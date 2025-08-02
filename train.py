"""
Production Training Script for CSM
Designed for scale: multi-GPU, mixed precision, gradient checkpointing
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import argparse
import yaml
import time
from typing import Dict, Optional
import wandb
from tqdm import tqdm

from architecture import CSMModel, conversational_loss
from rvq_tokenizer import ConversationalRVQ
from dataset_pipeline import ConversationalDataset, ConversationalCollator
from evaluation_framework import ComprehensiveConversationalEvaluator
from missing_components import (
    TrainingConfig, DistributedTrainingHelper,
    ExperimentTracker, generate_test_batch
)


class CSMTrainer:
    """Production trainer for CSM"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_distributed()
        self.setup_model()
        self.setup_data()
        self.setup_optimization()
        self.setup_logging()
        
        # Experiment tracking
        self.tracker = ExperimentTracker(config)
        self.best_val_score = 0.0
        
    def setup_distributed(self):
        """Setup distributed training"""
        self.is_distributed, self.rank, self.world_size, self.gpu = \
            DistributedTrainingHelper.setup_distributed()
        
        if self.is_distributed:
            self.device = torch.device(f'cuda:{self.gpu}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.is_main_process = self.rank == 0
        
    def setup_model(self):
        """Initialize models"""
        if self.is_main_process:
            print(f"Initializing {self.config.model_size} model...")
        
        # CSM model
        self.model = CSMModel(
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            max_seq_len=2048,
            rvq_codebooks=self.config.n_codebooks,
            rvq_codebook_size=self.config.codebook_size
        ).to(self.device)
        
        # RVQ tokenizer
        self.rvq = ConversationalRVQ(
            n_codebooks=self.config.n_codebooks,
            codebook_size=self.config.codebook_size,
            semantic_codebooks=self.config.semantic_codebooks
        ).to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.model_size != "tiny":
            self.model.gradient_checkpointing_enable()
        
        # Wrap in DDP
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.gpu],
                output_device=self.gpu,
                find_unused_parameters=False
            )
            self.rvq = DDP(
                self.rvq,
                device_ids=[self.gpu],
                output_device=self.gpu
            )
        
        # Evaluator (only on main process)
        if self.is_main_process:
            self.evaluator = ComprehensiveConversationalEvaluator()
        
        # Count parameters
        if self.is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters()) / 1e9
            print(f"Model parameters: {total_params:.2f}B")
    
    def setup_data(self):
        """Setup data loaders"""
        # Training data
        train_dataset = ConversationalDataset(
            data_paths=[f"{self.config.data_path}/train"],
            sample_rate=24000,
            max_duration=10.0,
            context_window=self.config.context_window,
            augment=True
        )
        
        # Distributed sampler
        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
        else:
            self.train_sampler = None
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            num_workers=4,
            pin_memory=True,
            collate_fn=ConversationalCollator()
        )
        
        # Validation data (only on main process)
        if self.is_main_process:
            val_dataset = ConversationalDataset(
                data_paths=[f"{self.config.data_path}/val"],
                sample_rate=24000,
                max_duration=10.0,
                context_window=self.config.context_window,
                augment=False
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=ConversationalCollator()
            )
    
    def setup_optimization(self):
        """Setup optimizer and scheduler"""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.rvq.parameters()),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler()
        
        # Learning rate scheduler
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                progress = (step - self.config.warmup_steps) / (
                    self.config.max_steps - self.config.warmup_steps
                )
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda
        )
        
        self.step = 0
        
    def setup_logging(self):
        """Setup logging and checkpointing"""
        if self.is_main_process:
            # Wandb
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=self.config.__dict__
            )
            
            # Checkpoint directory
            self.checkpoint_dir = Path(self.config.checkpoint_path) / self.config.experiment_name
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self):
        """Main training loop"""
        if self.is_main_process:
            print("Starting training...")
        
        self.model.train()
        self.rvq.train()
        
        # Progress bar
        if self.is_main_process:
            pbar = tqdm(total=self.config.max_steps, desc="Training")
        
        while self.step < self.config.max_steps:
            if self.is_distributed:
                self.train_sampler.set_epoch(self.step // len(self.train_loader))
            
            for batch in self.train_loader:
                loss = self.train_step(batch)
                
                if self.is_main_process:
                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Validation
                if self.step % 1000 == 0 and self.is_main_process:
                    self.validate()
                
                # Checkpoint
                if self.step % 5000 == 0:
                    self.save_checkpoint()
                
                self.step += 1
                
                if self.step >= self.config.max_steps:
                    break
        
        if self.is_main_process:
            pbar.close()
            print("Training complete!")
            
        # Final checkpoint
        self.save_checkpoint(final=True)
        
        # Cleanup
        if self.is_distributed:
            DistributedTrainingHelper.cleanup_distributed()
    
    def train_step(self, batch: Dict) -> float:
        """Single training step"""
        # Move batch to device
        batch = self._move_batch_to_device(batch)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            # Encode audio with RVQ
            codes, rvq_info = self.rvq.module.encode_conversational(
                batch['audio']
            ) if self.is_distributed else self.rvq.encode_conversational(batch['audio'])
            
            # Forward through CSM
            outputs = self.model(
                text_tokens=batch.get('text_tokens'),
                voice_codes=codes,
                conversation_history=batch.get('context')
            )
            
            # Add RVQ info
            outputs['rvq_loss'] = rvq_info['losses']['total']
            
            # Compute loss
            targets = self._prepare_targets(batch, codes)
            loss = conversational_loss(outputs, targets, context=batch)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.rvq.parameters()),
                self.config.gradient_clip_val
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
        
        # Logging
        if self.is_main_process and self.step % 10 == 0:
            wandb.log({
                'train/loss': loss.item() * self.config.gradient_accumulation_steps,
                'train/lr': self.scheduler.get_last_lr()[0],
                'train/step': self.step
            })
        
        # Hypothesis tracking
        if self.step % 100 == 0:
            self._track_hypotheses(outputs, batch)
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def validate(self):
        """Validation loop"""
        if not self.is_main_process:
            return
        
        print("\nRunning validation...")
        self.model.eval()
        self.rvq.eval()
        
        total_metrics = None
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                codes, rvq_info = self.rvq.encode_conversational(batch['audio'])
                outputs = self.model(
                    text_tokens=batch.get('text_tokens'),
                    voice_codes=codes
                )
                
                # Decode audio
                generated_audio = self.rvq.decode(codes)
                
                # Evaluate
                metrics = self.evaluator.evaluate_comprehensive(
                    generated={
                        'audio': generated_audio[0].cpu().numpy(),
                        'sr': 24000,
                        'text': batch['text'][0],
                        'speaker_id': batch['speaker_ids'][0]
                    },
                    reference={
                        'audio': batch['audio'][0].cpu().numpy(),
                        'sr': 24000,
                        'text': batch['text'][0]
                    },
                    context={'history': []}
                )
                
                # Accumulate metrics
                if total_metrics is None:
                    total_metrics = {k: 0.0 for k in metrics.__dict__.keys()}
                
                for key, value in metrics.__dict__.items():
                    if isinstance(value, (int, float)):
                        total_metrics[key] += value
                
                num_batches += 1
                
                if num_batches >= 50:  # Limit validation size
                    break
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        # Log metrics
        wandb.log({
            f'val/{k}': v for k, v in total_metrics.items()
            if isinstance(v, (int, float))
        })
        
        # Update best score
        val_score = total_metrics.get('character_consistency', 0.0)
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            self.save_checkpoint(best=True)
        
        print(f"Validation complete. Character consistency: {val_score:.3f}")
        
        self.model.train()
        self.rvq.train()
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to GPU"""
        batch['audio'] = batch['audio'].to(self.device)
        if 'text_tokens' in batch:
            batch['text_tokens'] = batch['text_tokens'].to(self.device)
        if 'audio_lengths' in batch:
            batch['audio_lengths'] = batch['audio_lengths'].to(self.device)
        return batch
    
    def _prepare_targets(self, batch: Dict, codes: List[torch.Tensor]) -> Dict:
        """Prepare targets for loss computation"""
        targets = {}
        
        # Text targets (shifted by 1)
        if 'text_tokens' in batch:
            targets['text_targets'] = batch['text_tokens'][:, 1:]
        
        # Voice targets (the actual codes)
        targets['voice_targets'] = codes
        
        return targets
    
    def _track_hypotheses(self, outputs: Dict, batch: Dict):
        """Track hypothesis validation metrics"""
        # This would implement the hypothesis tracking from proof_of_concept.py
        pass
    
    def save_checkpoint(self, best: bool = False, final: bool = False):
        """Save model checkpoint"""
        if not self.is_main_process:
            return
        
        if best:
            path = self.checkpoint_dir / "best_model.pt"
        elif final:
            path = self.checkpoint_dir / "final_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_step_{self.step}.pt"
        
        # Prepare checkpoint
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            'rvq_state_dict': self.rvq.module.state_dict() if self.is_distributed else self.rvq.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'best_val_score': self.best_val_score
        }
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
            self.rvq.module.load_state_dict(checkpoint['rvq_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.rvq.load_state_dict(checkpoint['rvq_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.best_val_score = checkpoint.get('best_val_score', 0.0)
        
        print(f"Loaded checkpoint from {path} (step {self.step})")


def main():
    parser = argparse.ArgumentParser(description='Train CSM model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--model_size', type=str, default='tiny',
                        choices=['tiny', 'small', 'medium'],
                        help='Model size')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode with synthetic data')
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    # Override model size if specified
    if args.model_size:
        config.model_size = args.model_size
        config.scale_config()
    
    # Create trainer
    trainer = CSMTrainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Debug mode
    if args.debug:
        print("Running in debug mode with synthetic data...")
        for i in range(10):
            batch = generate_test_batch(config)
            loss = trainer.train_step(batch)
            print(f"Step {i}: Loss = {loss:.4f}")
        return
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()