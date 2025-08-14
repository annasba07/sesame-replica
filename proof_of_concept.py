"""
Minimal Proof of Concept
Goal: Validate that voice-text fusion creates better conversational AI
Approach: Train tiny model (350M) on 1k hours, test key hypotheses
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path

from architecture import CSMModel, conversational_loss
from rvq_tokenizer import ConversationalRVQ
from dataset_pipeline import create_dataloader
from evaluation_framework import ComprehensiveConversationalEvaluator


class CSMLightning(pl.LightningModule):
    """Lightning wrapper for rapid experimentation"""
    
    def __init__(
        self,
        model_size: str = "tiny",  # tiny, small, medium
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        validate_hypotheses: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model configurations
        configs = {
            "tiny": {"d_model": 768, "n_layers": 12, "n_heads": 12},      # 350M
            "small": {"d_model": 1536, "n_layers": 16, "n_heads": 16},    # 1.5B
            "medium": {"d_model": 2048, "n_layers": 24, "n_heads": 16}    # 3B
        }
        
        config = configs[model_size]
        
        # Initialize models
        self.csm = CSMModel(
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            max_seq_len=2048
        )
        
        self.rvq = ConversationalRVQ(
            n_codebooks=32,
            codebook_size=1024,
            semantic_codebooks=10
        )
        
        # Evaluator for validation
        self.evaluator = ComprehensiveConversationalEvaluator()
        
        # Hypothesis tracking
        self.hypothesis_results = {
            "voice_semantic_coupling": [],
            "prosodic_memory_benefit": [],
            "character_consistency": []
        }
        
    def forward(self, batch: Dict) -> Dict:
        # Extract RVQ codes from audio
        audio = batch['audio']
        codes, rvq_info = self.rvq.encode_conversational(audio)
        
        # Forward through CSM
        outputs = self.csm(
            text_tokens=batch.get('text_tokens'),
            voice_codes=codes,
            conversation_history=batch.get('context')
        )
        
        outputs['rvq_info'] = rvq_info
        return outputs
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch)
        
        # Compute loss
        targets = {
            'text_targets': batch.get('text_tokens_target'),
            'voice_targets': batch.get('voice_codes_target')
        }
        
        loss = conversational_loss(outputs, targets, context=batch.get('context'))
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('rvq_commitment_loss', outputs['rvq_info']['losses']['commitment'])
        
        # Log hypothesis metrics
        if self.hparams.validate_hypotheses and batch_idx % 100 == 0:
            self._validate_hypotheses(outputs, batch)
        
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        outputs = self.forward(batch)
        
        # Generate audio from codes
        generated_audio = self.rvq.decode(outputs['voice_logits'])
        
        # Evaluate
        metrics = self.evaluator.evaluate_comprehensive(
            generated={
                'audio': generated_audio[0].cpu().numpy(),
                'sr': 24000,
                'text': self._decode_text(outputs['text_logits'][0]),
                'speaker_id': batch['speaker_ids'][0]
            },
            reference={
                'audio': batch['audio'][0].cpu().numpy(),
                'sr': 24000,
                'text': batch['text'][0]
            },
            context={'history': batch.get('context', [])}
        )
        
        # Log all metrics
        for field in metrics.__dataclass_fields__:
            value = getattr(metrics, field)
            if isinstance(value, (int, float)):
                self.log(f'val_{field}', value)
        
        return {'metrics': metrics}
    
    def _validate_hypotheses(self, outputs: Dict, batch: Dict):
        """Test our key hypotheses during training"""
        
        # Hypothesis 1: Voice tokens contain semantic information
        if 'voice_logits' in outputs and 'text_tokens_target' in batch:
            # Can we predict next word from voice codes alone?
            voice_features = outputs['rvq_info']['semantic']  # Semantic features from RVQ
            
            # Simple probe: linear layer from voice to text
            with torch.no_grad():
                probe = nn.Linear(voice_features.size(-1), self.csm.vocab_size).to(self.device)
                text_pred_from_voice = probe(voice_features.mean(dim=2))
                
                # Measure accuracy
                text_targets = batch['text_tokens_target']
                accuracy = (text_pred_from_voice.argmax(-1) == text_targets).float().mean()
                
                self.hypothesis_results['voice_semantic_coupling'].append(accuracy.item())
                self.log('hypothesis/voice_semantic_coupling', accuracy)
        
        # Hypothesis 2: Prosodic memory improves coherence
        # Compare model with and without memory updates
        if hasattr(self.csm, 'memory'):
            # Forward without memory
            self.csm.memory.eval()  # Disable updates
            outputs_no_mem = self.csm(
                text_tokens=batch.get('text_tokens'),
                voice_codes=outputs['rvq_info']['codes']
            )
            self.csm.memory.train()
            
            # Compare perplexity
            loss_with_mem = conversational_loss(outputs, batch)
            loss_without_mem = conversational_loss(outputs_no_mem, batch)
            
            improvement = (loss_without_mem - loss_with_mem) / loss_without_mem
            self.hypothesis_results['prosodic_memory_benefit'].append(improvement.item())
            self.log('hypothesis/memory_benefit', improvement)
    
    def _decode_text(self, logits: torch.Tensor) -> str:
        """Decode text from logits"""
        tokens = logits.argmax(dim=-1)
        # Would use proper tokenizer decode
        return f"decoded_text_{tokens[0].item()}"
    
    def configure_optimizers(self):
        # Use AdamW with cosine schedule
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / self.hparams.warmup_steps
            else:
                progress = (step - self.hparams.warmup_steps) / (
                    self.trainer.max_steps - self.hparams.warmup_steps
                )
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


def run_proof_of_concept():
    """Run the proof of concept experiment"""
    
    # Initialize wandb for experiment tracking
    wandb.init(project="csm-proof-of-concept", name="tiny-1k-hours")
    
    # Create model
    model = CSMLightning(
        model_size="tiny",
        learning_rate=3e-4,
        validate_hypotheses=True
    )
    
    # Create dataloaders
    # For PoC, we'd use a subset of LibriSpeech or similar
    train_loader = create_dataloader(
        ["path/to/train/data"],
        batch_size=32,
        num_workers=4,
        augment=True
    )
    
    val_loader = create_dataloader(
        ["path/to/val/data"],
        batch_size=16,
        num_workers=4,
        augment=False
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_steps=50000,  # ~1 week of training
        val_check_interval=1000,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,  # Effective batch size 128
        precision=16,  # Mixed precision
        devices=torch.cuda.device_count(),
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="csm-{step}-{val_character_consistency:.3f}",
                save_top_k=3,
                monitor="val_character_consistency",
                mode="max"
            ),
            EarlyStopping(
                monitor="val_character_consistency",
                patience=5,
                mode="max"
            )
        ],
        logger=WandbLogger()
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Analyze hypothesis results
    print("\n=== HYPOTHESIS VALIDATION RESULTS ===")
    print(f"Voice-Semantic Coupling: {np.mean(model.hypothesis_results['voice_semantic_coupling']):.3f}")
    print(f"Prosodic Memory Benefit: {np.mean(model.hypothesis_results['prosodic_memory_benefit']):.3f}")
    
    return model


def run_ablation_studies(base_model: CSMLightning):
    """Run ablation studies to validate design choices"""
    
    ablations = {
        "no_cross_attention": "Disable cross-modal attention",
        "no_memory": "Remove conversational memory",
        "no_semantic_rvq": "Use uniform RVQ without semantic bias",
        "no_prosody": "Remove prosodic features",
        "symmetric_attention": "Use same attention for text and voice"
    }
    
    results = {}
    
    for ablation_name, description in ablations.items():
        print(f"\nRunning ablation: {description}")
        
        # Create ablated model
        ablated_model = create_ablated_model(base_model, ablation_name)
        
        # Quick training (10% of full)
        trainer = pl.Trainer(
            max_steps=5000,
            val_check_interval=500,
            devices=1,
            logger=WandbLogger(name=f"ablation-{ablation_name}")
        )
        
        # Use same data
        train_loader = create_dataloader(["path/to/train/data"], batch_size=32)
        val_loader = create_dataloader(["path/to/val/data"], batch_size=16)
        
        trainer.fit(ablated_model, train_loader, val_loader)
        
        # Compare to baseline
        results[ablation_name] = {
            'final_loss': trainer.callback_metrics['val_loss'].item(),
            'character_consistency': trainer.callback_metrics.get('val_character_consistency', 0).item()
        }
    
    return results


def create_ablated_model(base_model: CSMLightning, ablation_type: str) -> CSMLightning:
    """Create model with specific ablation"""
    
    # Clone base model
    import copy
    model = copy.deepcopy(base_model)
    
    if ablation_type == "no_cross_attention":
        # Replace cross attention with self attention
        for layer in model.csm.layers:
            layer['cross_attn'] = nn.Identity()
    
    elif ablation_type == "no_memory":
        # Disable memory
        model.csm.memory = nn.Identity()
    
    elif ablation_type == "no_semantic_rvq":
        # Set semantic codebooks to 0
        model.rvq.semantic_codebooks = 0
    
    # ... implement other ablations
    
    return model


if __name__ == "__main__":
    # Run proof of concept
    model = run_proof_of_concept()
    
    # Run ablations
    ablation_results = run_ablation_studies(model)
    
    print("\n=== ABLATION STUDY RESULTS ===")
    for name, results in ablation_results.items():
        print(f"{name}: Loss={results['final_loss']:.3f}, Consistency={results['character_consistency']:.3f}")