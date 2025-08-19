#!/usr/bin/env python3
"""
Production Training Script for CSM
Designed for scalability and monitoring
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import our modules
from architecture import CSMModel, conversational_loss
from rvq_tokenizer import ConversationalRVQ
from dataset_pipeline import ConversationalDataset, ConversationalCollator
from evaluation_framework import ComprehensiveConversationalEvaluator


class CSMTrainer:
    """Production-ready trainer with all best practices"""
    
    def __init__(self, config: Dict, resume_from: Optional[str] = None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(f"experiments/{config['name']}_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.exp_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize components
        self._init_models()
        self._init_data()
        self._init_training()
        self._init_logging()
        
        # Resume if needed
        if resume_from:
            self.resume(resume_from)
    
    def _init_models(self):
        """Initialize models"""
        print(f"Initializing models on {self.device}...")
        
        # CSM Model
        self.model = CSMModel(
            **self.config['model']
        ).to(self.device)
        
        # RVQ Tokenizer
        self.rvq = ConversationalRVQ(
            **self.config['rvq']
        ).to(self.device)
        
        # Count parameters
        model_params = sum(p.numel() for p in self.model.parameters())
        rvq_params = sum(p.numel() for p in self.rvq.parameters())
        print(f"Model parameters: {model_params:,}")
        print(f"RVQ parameters: {rvq_params:,}")
        print(f"Total parameters: {model_params + rvq_params:,}")
        
        # Multi-GPU
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            self.rvq = nn.DataParallel(self.rvq)
    
    def _init_data(self):
        """Initialize data loaders"""
        print("Initializing data loaders...")
        
        # Load splits
        data_dir = Path(self.config['data']['data_dir'])
        with open(data_dir / "splits.json", 'r') as f:
            splits = json.load(f)
        
        # Create datasets
        self.train_dataset = ConversationalDataset(
            data_dir,
            split='train',
            sample_rate=self.config['data']['sample_rate'],
            augment=True
        )
        
        self.val_dataset = ConversationalDataset(
            data_dir,
            split='val',
            sample_rate=self.config['data']['sample_rate'],
            augment=False
        )
        
        # Create loaders
        collator = ConversationalCollator(
            max_audio_length=self.config['data']['max_audio_length']
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            collate_fn=collator,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            collate_fn=collator,
            pin_memory=True
        )
        
        print(f"Train samples: {len(self.train_dataset):,}")
        print(f"Val samples: {len(self.val_dataset):,}")
    
    def _init_training(self):
        """Initialize training components"""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.rvq.parameters()),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['max_epochs'] * len(self.train_loader),
            eta_min=self.config['training']['learning_rate'] * 0.01
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config['training']['mixed_precision'] else None
        
        # Evaluator
        self.evaluator = ComprehensiveConversationalEvaluator()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _init_logging(self):
        """Initialize logging"""
        self.writer = SummaryWriter(self.exp_dir / "tensorboard")
        self.log_file = open(self.exp_dir / "training.log", 'w')
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.rvq.train()
        
        epoch_losses = []
        epoch_metrics = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            with torch.cuda.amp.autocast() if self.scaler else torch.no_grad():
                # Encode audio
                codes, rvq_info = self.rvq.encode(batch['audio'])
                
                # Model forward
                outputs = self.model(
                    text_tokens=batch.get('text_tokens'),
                    voice_codes=codes
                )
                
                # Compute loss
                targets = {
                    'text_targets': batch.get('text_tokens'),
                    'voice_targets': codes
                }
                loss = conversational_loss(outputs, targets)
                
                # Add RVQ losses
                if 'losses' in rvq_info:
                    loss = loss + rvq_info['losses']['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.rvq.parameters()),
                    self.config['training']['grad_clip']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.rvq.parameters()),
                    self.config['training']['grad_clip']
                )
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Logging
            epoch_losses.append(loss.item())
            self.global_step += 1
            
            if self.global_step % self.config['logging']['log_interval'] == 0:
                self._log_step(loss.item(), batch_idx)
            
            if self.global_step % self.config['logging']['eval_interval'] == 0:
                val_metrics = self.evaluate()
                self._log_validation(val_metrics)
                self.model.train()
                self.rvq.train()
        
        return {
            'loss': sum(epoch_losses) / len(epoch_losses),
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        self.rvq.eval()
        
        val_losses = []
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch)
                
                # Forward
                codes, _ = self.rvq.encode(batch['audio'])
                outputs = self.model(
                    text_tokens=batch.get('text_tokens'),
                    voice_codes=codes
                )
                
                # Loss
                targets = {
                    'text_targets': batch.get('text_tokens'),
                    'voice_targets': codes
                }
                loss = conversational_loss(outputs, targets)
                
                val_losses.append(loss.item())
                all_outputs.append(outputs)
                all_targets.append(targets)
        
        # Compute metrics
        metrics = {
            'val_loss': sum(val_losses) / len(val_losses)
        }
        
        # Additional evaluation metrics
        if self.epoch % 5 == 0:  # Full evaluation every 5 epochs
            eval_results = self.evaluator.evaluate_batch(all_outputs, all_targets)
            metrics.update(eval_results)
        
        return metrics
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config['training']['max_epochs']} epochs...")
        
        for epoch in range(self.epoch, self.config['training']['max_epochs']):
            self.epoch = epoch
            print(f"\nEpoch {epoch+1}/{self.config['training']['max_epochs']}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.evaluate()
            
            # Log epoch
            self._log_epoch(train_metrics, val_metrics)
            
            # Save checkpoint
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint('best_model.pt')
            
            if (epoch + 1) % self.config['logging']['checkpoint_interval'] == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pt')
        
        print("\nTraining complete!")
        self._cleanup()
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'rvq_state': self.rvq.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state'] = self.scaler.state_dict()
        
        path = self.exp_dir / "checkpoints" / filename
        path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def resume(self, checkpoint_path: str):
        """Resume from checkpoint"""
        print(f"Resuming from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.rvq.load_state_dict(checkpoint['rvq_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        self.epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler and 'scaler_state' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state'])
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device"""
        batch['audio'] = batch['audio'].to(self.device)
        if 'text_tokens' in batch:
            batch['text_tokens'] = batch['text_tokens'].to(self.device)
        return batch
    
    def _log_step(self, loss: float, batch_idx: int):
        """Log training step"""
        # Tensorboard
        self.writer.add_scalar('train/loss', loss, self.global_step)
        self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
        
        # Console
        if batch_idx % 10 == 0:
            print(f"  Step {self.global_step}: loss={loss:.4f}, lr={self.scheduler.get_last_lr()[0]:.2e}")
    
    def _log_validation(self, metrics: Dict[str, float]):
        """Log validation metrics"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'val/{key}', value, self.global_step)
    
    def _log_epoch(self, train_metrics: Dict, val_metrics: Dict):
        """Log epoch summary"""
        log_str = f"Epoch {self.epoch+1}: "
        log_str += f"train_loss={train_metrics['loss']:.4f}, "
        log_str += f"val_loss={val_metrics['val_loss']:.4f}, "
        log_str += f"lr={train_metrics['lr']:.2e}"
        
        print(log_str)
        self.log_file.write(log_str + "\n")
        self.log_file.flush()
    
    def _cleanup(self):
        """Cleanup resources"""
        self.writer.close()
        self.log_file.close()


def main():
    parser = argparse.ArgumentParser(description="Train CSM model")
    parser.add_argument('--config', type=str, default='configs/small_gpu_config.json',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Override max epochs from config')
    parser.add_argument('--name', type=str, default='csm',
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override settings
    config['name'] = args.name
    if args.max_epochs:
        config['training']['max_epochs'] = args.max_epochs
    
    # Add default values
    config.setdefault('data', {})
    config['data'].setdefault('data_dir', 'data/conversations')
    config['data'].setdefault('sample_rate', 24000)
    config['data'].setdefault('max_audio_length', 240000)
    config['data'].setdefault('num_workers', 4)
    
    config['training'].setdefault('weight_decay', 0.01)
    config['training'].setdefault('grad_clip', 1.0)
    config['training'].setdefault('mixed_precision', True)
    config['training'].setdefault('max_epochs', 100)
    
    config.setdefault('logging', {})
    config['logging'].setdefault('log_interval', 10)
    config['logging'].setdefault('eval_interval', 500)
    config['logging'].setdefault('checkpoint_interval', 5)
    
    # Create trainer
    trainer = CSMTrainer(config, resume_from=args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()