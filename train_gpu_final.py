#!/usr/bin/env python3
"""
Final GPU Training Script - Full CSM Model
"""

import torch
import torch.nn as nn
import torch.amp as amp
import json
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

def main():
    # Import our modules
    from architecture import CSMModel, conversational_loss
    from rvq_tokenizer import ConversationalRVQ
    from missing_components import generate_test_batch, TrainingConfig
    
    print("GPU Training Script (Final)")
    print("=" * 50)
    
    # Check GPU
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Production config - optimized for RTX 3070 Ti
    config = {
        'model': {
            'd_model': 512,      # Production size
            'n_layers': 6,       # Good depth
            'n_heads': 8,        # Efficient attention
            'max_seq_len': 2048  # Good context
        },
        'rvq': {
            'n_codebooks': 8,    # Full hierarchical RVQ
            'codebook_size': 256,
            'latent_dim': 128
        },
        'training': {
            'batch_size': 4,     # Fits in 8GB
            'learning_rate': 3e-4,
            'gradient_accumulation': 4,  # Effective batch = 16
            'num_epochs': 1,     # Quick demo
            'steps_per_epoch': 100
        }
    }
    
    # Create models
    print("\nCreating models...")
    model = CSMModel(**config['model']).to(device)
    rvq = ConversationalRVQ(**config['rvq']).to(device)
    
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in rvq.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(rvq.parameters()),
        lr=config['training']['learning_rate'],
        betas=(0.9, 0.95)
    )
    
    # Mixed precision
    scaler = amp.GradScaler('cuda')
    
    # Training config
    test_config = TrainingConfig(
        batch_size=config['training']['batch_size'],
        max_audio_length=24000,  # 1 second for testing
        d_model=config['model']['d_model']
    )
    
    # Training
    print(f"\nStarting training...")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Steps per epoch: {config['training']['steps_per_epoch']}")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation']}")
    
    model.train()
    rvq.train()
    
    training_history = []
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        epoch_losses = []
        epoch_start = time.time()
        
        progress_bar = tqdm(range(config['training']['steps_per_epoch']), 
                          desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for step in progress_bar:
            # Generate batch
            batch = generate_test_batch(test_config)
            batch['audio'] = batch['audio'].to(device)
            if 'text_tokens' in batch:
                batch['text_tokens'] = batch['text_tokens'].to(device)
            
            # Forward pass with mixed precision
            with amp.autocast('cuda'):
                # Encode audio
                codes, rvq_info = rvq.encode(batch['audio'])
                
                # Model forward
                outputs = model(text_tokens=batch.get('text_tokens'), voice_codes=codes)
                
                # Compute loss
                targets = {
                    'text_targets': batch.get('text_tokens'),
                    'voice_targets': codes
                }
                loss = conversational_loss(outputs, targets)
                
                # Add RVQ loss
                if 'losses' in rvq_info:
                    loss = loss + 0.1 * rvq_info['losses']['total']
                
                # Scale for gradient accumulation
                loss = loss / config['training']['gradient_accumulation']
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Update weights
            if (step + 1) % config['training']['gradient_accumulation'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(rvq.parameters()), 
                    max_norm=1.0
                )
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                global_step += 1
            
            # Record loss
            epoch_losses.append(loss.item() * config['training']['gradient_accumulation'])
            
            # Update progress
            if len(epoch_losses) > 0:
                avg_loss = sum(epoch_losses[-10:]) / len(epoch_losses[-10:])
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Clear cache periodically
            if step % 20 == 0:
                torch.cuda.empty_cache()
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average loss: {avg_epoch_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_epoch_loss,
            'time': epoch_time
        })
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    rvq.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            test_text = torch.randint(0, 1000, (1, 20)).to(device)
            with amp.autocast('cuda'):
                _ = model(text_tokens=test_text)
    
    # Measure latency
    latencies = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            with amp.autocast('cuda'):
                test_text = torch.randint(0, 1000, (1, 20)).to(device)
                outputs = model(text_tokens=test_text)
        
        torch.cuda.synchronize()
        latency = (time.time() - start) * 1000
        latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies)
    
    # Save checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'model_state': model.state_dict(),
        'rvq_state': rvq.state_dict(),
        'config': config,
        'training_history': training_history,
        'best_loss': best_loss,
        'avg_latency_ms': avg_latency,
        'total_params': total_params,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = checkpoint_dir / 'csm_gpu_final.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Model parameters: {total_params/1e6:.1f}M")
    print(f"Final loss: {training_history[-1]['loss']:.4f}")
    print(f"Average inference latency: {avg_latency:.1f}ms")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"\nModel saved to: {checkpoint_path}")
    
    # Check if we achieved target
    if avg_latency < 200:
        print(f"\nSUCCESS: Achieved {avg_latency:.1f}ms latency (< 200ms target!)")
    else:
        print(f"\nWARNING: Latency {avg_latency:.1f}ms is above 200ms target")
    
    return checkpoint_path, avg_latency

if __name__ == "__main__":
    try:
        checkpoint_path, latency = main()
        
        print("\nNext steps:")
        print(f"1. Run full benchmarks: python benchmark.py --checkpoint {checkpoint_path}")
        print(f"2. Deploy API: python serve_api.py --checkpoint {checkpoint_path}")
        print(f"3. Create demo: python interactive_demo.py --checkpoint {checkpoint_path}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Try reducing batch_size or model size in the config")