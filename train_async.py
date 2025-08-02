#!/usr/bin/env python3
"""Async training script that runs in background"""

import torch
import json
from pathlib import Path
import time
import threading
import queue
from datetime import datetime

# Import our modules
from architecture import CSMModel, conversational_loss
from rvq_tokenizer import ConversationalRVQ
from missing_components import generate_test_batch, TrainingConfig

class AsyncTrainer:
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = torch.device(device)
        self.running = False
        self.progress_queue = queue.Queue()
        
        # Create models
        self.model = CSMModel(
            d_model=config['model']['d_model'],
            n_layers=config['model']['n_layers'],
            n_heads=config['model']['n_heads']
        ).to(self.device)
        
        self.rvq = ConversationalRVQ(
            n_codebooks=config['rvq']['n_codebooks'],
            codebook_size=config['rvq']['codebook_size']
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.rvq.parameters()),
            lr=config['training']['learning_rate']
        )
        
        self.losses = []
        self.start_time = None
        
    def train_step(self, batch):
        """Single training step"""
        # Move to device
        batch['audio'] = batch['audio'].to(self.device)
        if 'text_tokens' in batch:
            batch['text_tokens'] = batch['text_tokens'].to(self.device)
        
        # Forward pass
        codes, rvq_info = self.rvq.encode(batch['audio'])
        outputs = self.model(text_tokens=batch.get('text_tokens'), voice_codes=codes)
        
        # Loss
        targets = {
            'text_targets': batch.get('text_tokens'),
            'voice_targets': codes
        }
        loss = conversational_loss(outputs, targets)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def training_loop(self, num_steps=100):
        """Background training loop"""
        self.model.train()
        self.rvq.train()
        
        test_config = TrainingConfig()
        
        for step in range(num_steps):
            if not self.running:
                break
                
            # Generate batch
            batch = generate_test_batch(test_config)
            
            # Train
            loss = self.train_step(batch)
            self.losses.append(loss)
            
            # Report progress
            progress = {
                'step': step,
                'loss': loss,
                'elapsed': time.time() - self.start_time,
                'timestamp': datetime.now().isoformat()
            }
            self.progress_queue.put(progress)
            
            # Save checkpoint every 10 steps
            if (step + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_step_{step+1}.pt')
        
        self.running = False
    
    def start(self, num_steps=100):
        """Start async training"""
        if self.running:
            return False
            
        self.running = True
        self.start_time = time.time()
        
        # Start training in background thread
        self.thread = threading.Thread(
            target=self.training_loop,
            args=(num_steps,),
            daemon=True
        )
        self.thread.start()
        
        return True
    
    def stop(self):
        """Stop training"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=5)
    
    def get_progress(self):
        """Get training progress"""
        updates = []
        while not self.progress_queue.empty():
            updates.append(self.progress_queue.get())
        return updates
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'rvq_state': self.rvq.state_dict(),
            'config': self.config,
            'losses': self.losses,
            'step': len(self.losses)
        }, f'checkpoints/{filename}')


def main():
    """Example usage"""
    print("Starting async training...")
    
    # Config
    config = {
        'model': {'d_model': 256, 'n_layers': 4, 'n_heads': 4},
        'rvq': {'n_codebooks': 8, 'codebook_size': 256},
        'training': {'batch_size': 1, 'learning_rate': 3e-4}
    }
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = AsyncTrainer(config, device)
    
    # Start training
    trainer.start(num_steps=50)
    print("Training started in background...")
    
    # Monitor progress
    try:
        while trainer.running:
            time.sleep(2)  # Check every 2 seconds
            
            # Get updates
            updates = trainer.get_progress()
            for update in updates:
                print(f"Step {update['step']}: loss={update['loss']:.4f}, time={update['elapsed']:.1f}s")
            
    except KeyboardInterrupt:
        print("\nStopping training...")
        trainer.stop()
    
    # Final stats
    if trainer.losses:
        print(f"\nTraining complete!")
        print(f"Steps: {len(trainer.losses)}")
        print(f"Initial loss: {trainer.losses[0]:.4f}")
        print(f"Final loss: {trainer.losses[-1]:.4f}")
        print(f"Improvement: {(trainer.losses[0] - trainer.losses[-1]) / trainer.losses[0] * 100:.1f}%")
        
        # Save final checkpoint
        trainer.save_checkpoint('final_model.pt')
        print("Saved final checkpoint")


if __name__ == "__main__":
    main()