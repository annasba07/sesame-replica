#!/usr/bin/env python3
"""
GPU Training Script - Optimized for RTX 3070 Ti (8GB)
"""

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import json
import time
from pathlib import Path
from tqdm import tqdm

# Import our modules
from architecture import CSMModel, conversational_loss
from rvq_tokenizer import ConversationalRVQ
from dataset_pipeline import create_dataloader
from missing_components import generate_test_batch, TrainingConfig

print("GPU Training Script")
print("=" * 50)

# Check GPU
device = torch.device('cuda')
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Optimized config for RTX 3070 Ti (8GB)
config = {
    'model': {
        'd_model': 512,      # Medium size that fits in 8GB
        'n_layers': 8,       # Reasonable depth
        'n_heads': 8,        # Good parallelization
        'max_seq_len': 2048
    },
    'rvq': {
        'n_codebooks': 16,   # Reduced for memory
        'codebook_size': 512,
        'latent_dim': 256
    },
    'training': {
        'batch_size': 4,     # Small batch for 8GB
        'learning_rate': 3e-4,
        'gradient_accumulation': 4,  # Effective batch size = 16
        'mixed_precision': True
    }
}

# Calculate model size
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Create models
print("\nCreating models...")
model = CSMModel(**config['model']).to(device)
rvq = ConversationalRVQ(**config['rvq']).to(device)

model_params = count_parameters(model)
rvq_params = count_parameters(rvq)
total_params = model_params + rvq_params

print(f"Model parameters: {model_params:,}")
print(f"RVQ parameters: {rvq_params:,}")
print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

# Memory estimate
param_memory = total_params * 4 / 1e9  # FP32
print(f"Parameter memory: {param_memory:.2f} GB")
print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9 - param_memory:.2f} GB")

# Create optimizer with memory-efficient settings
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(rvq.parameters()),
    lr=config['training']['learning_rate'],
    betas=(0.9, 0.95),  # Reduce Adam memory
    eps=1e-8
)

# Mixed precision for memory efficiency
scaler = amp.GradScaler()

# Try to load real data, fallback to synthetic
try:
    print("\nLoading data...")
    train_loader = create_dataloader(
        ["data/conversations"],
        batch_size=config['training']['batch_size'],
        num_workers=2  # Reduce for Windows
    )
    print(f"Loaded {len(train_loader)} batches of real data")
    use_real_data = True
except Exception as e:
    print(f"Could not load real data: {e}")
    print("Using synthetic data for demonstration")
    train_loader = None
    use_real_data = False

# Training settings
num_epochs = 3
steps_per_epoch = 100 if not use_real_data else len(train_loader)
gradient_accumulation_steps = config['training']['gradient_accumulation']

# Create checkpoint directory
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Training loop
print(f"\nStarting training for {num_epochs} epochs...")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Gradient accumulation: {gradient_accumulation_steps}")
print(f"Effective batch size: {config['training']['batch_size'] * gradient_accumulation_steps}")

model.train()
rvq.train()

global_step = 0
best_loss = float('inf')
training_history = []

for epoch in range(num_epochs):
    epoch_losses = []
    epoch_start = time.time()
    
    progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for step in progress_bar:
        # Get batch
        if use_real_data and train_loader:
            try:
                batch = next(iter(train_loader))
            except StopIteration:
                break
        else:
            test_config = TrainingConfig()
            batch = generate_test_batch(test_config)
        
        # Move to GPU
        batch['audio'] = batch['audio'].to(device)
        if 'text_tokens' in batch:
            batch['text_tokens'] = batch['text_tokens'].to(device)
        
        # Forward pass with mixed precision
        with amp.autocast():
            # Encode audio to voice codes
            codes, rvq_info = rvq.encode(batch['audio'])
            
            # Forward through model
            outputs = model(text_tokens=batch.get('text_tokens'), voice_codes=codes)
            
            # Compute loss
            targets = {
                'text_targets': batch.get('text_tokens'),
                'voice_targets': codes
            }
            loss = conversational_loss(outputs, targets)
            
            # Add RVQ reconstruction loss
            if 'losses' in rvq_info:
                loss = loss + 0.1 * rvq_info['losses']['total']
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every gradient_accumulation_steps
        if (step + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(rvq.parameters()), 
                max_norm=1.0
            )
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            global_step += 1
        
        # Record loss
        epoch_losses.append(loss.item() * gradient_accumulation_steps)
        
        # Update progress bar
        if len(epoch_losses) > 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Free memory periodically
        if step % 50 == 0:
            torch.cuda.empty_cache()
    
    # Epoch summary
    epoch_time = time.time() - epoch_start
    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Average loss: {avg_epoch_loss:.4f}")
    print(f"  Time: {epoch_time:.1f}s ({epoch_time/steps_per_epoch:.2f}s/step)")
    print(f"  GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    training_history.append({
        'epoch': epoch + 1,
        'loss': avg_epoch_loss,
        'time': epoch_time
    })
    
    # Save checkpoint
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'rvq_state': rvq.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'config': config,
            'loss': avg_epoch_loss,
            'training_history': training_history
        }
        
        checkpoint_path = checkpoint_dir / 'best_model_gpu.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved best model to {checkpoint_path}")

# Final evaluation
print("\n" + "=" * 50)
print("Training Complete!")
print("=" * 50)

# Test generation
print("\nTesting model generation...")
model.eval()
rvq.eval()

with torch.no_grad():
    # Generate from text
    test_text = torch.randint(0, 1000, (1, 10)).to(device)
    with amp.autocast():
        outputs = model(text_tokens=test_text)
    print(f"Generated voice logits shape: {outputs['voice_logits'].shape}")
    
    # Test latency
    torch.cuda.synchronize()
    start = time.time()
    
    with amp.autocast():
        _ = model(text_tokens=test_text)
    
    torch.cuda.synchronize()
    latency = (time.time() - start) * 1000
    print(f"Inference latency: {latency:.1f}ms")

# Save final checkpoint
final_checkpoint = {
    'model_state': model.state_dict(),
    'rvq_state': rvq.state_dict(),
    'config': config,
    'training_history': training_history,
    'device': str(device),
    'total_params': total_params
}

final_path = checkpoint_dir / 'csm_gpu_trained.pt'
torch.save(final_checkpoint, final_path)
print(f"\nFinal model saved to {final_path}")

# Training summary
print("\n" + "=" * 50)
print("TRAINING SUMMARY")
print("=" * 50)
print(f"Total parameters: {total_params/1e6:.1f}M")
print(f"Final loss: {training_history[-1]['loss']:.4f}")
print(f"Best loss: {best_loss:.4f}")
print(f"Total training time: {sum(h['time'] for h in training_history):.1f}s")
print(f"Inference latency: {latency:.1f}ms")
print(f"Device: {torch.cuda.get_device_name(0)}")

print("\nNext steps:")
print("1. Run benchmarks: python benchmark.py --checkpoint checkpoints/csm_gpu_trained.pt")
print("2. Deploy API: python serve_api.py --checkpoint checkpoints/csm_gpu_trained.pt")
print("3. Create demo: python interactive_demo.py --checkpoint checkpoints/csm_gpu_trained.pt")