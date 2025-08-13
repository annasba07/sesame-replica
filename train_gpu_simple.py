#!/usr/bin/env python3
"""
Simple GPU Training - No multiprocessing issues
"""

import torch
import torch.nn as nn
import torch.amp as amp
import json
import time
from pathlib import Path
from tqdm import tqdm

def main():
    # Import our modules
    from architecture import CSMModel, conversational_loss
    from rvq_tokenizer import ConversationalRVQ
    from missing_components import generate_test_batch, TrainingConfig

    print("GPU Training Script (Simple)")
    print("=" * 50)

    # Check GPU
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Smaller config to ensure it fits
    config = {
        'model': {
            'd_model': 128,      # Very small for memory
            'n_layers': 2,       
            'n_heads': 4,        
            'max_seq_len': 512
        },
        'rvq': {
            'n_codebooks': 4,   
            'codebook_size': 128,
            'latent_dim': 64
        },
        'training': {
            'batch_size': 2,     # Very small batch
            'learning_rate': 3e-4,
        }
    }

    # Clear cache before creating models
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Create models
    print("\nCreating models...")
    model = CSMModel(**config['model']).to(device)
    rvq = ConversationalRVQ(**config['rvq']).to(device)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in rvq.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(rvq.parameters()),
        lr=config['training']['learning_rate']
    )

    # Mixed precision
    scaler = amp.GradScaler('cuda')

    # Use synthetic data for quick training
    test_config = TrainingConfig()
    num_steps = 50  # Quick training

    # Training
    print(f"\nTraining for {num_steps} steps...")
    model.train()
    rvq.train()

    losses = []
    start_time = time.time()

    for step in tqdm(range(num_steps)):
        # Generate batch
        batch = generate_test_batch(test_config)
        batch['audio'] = batch['audio'].to(device)
        if 'text_tokens' in batch:
            batch['text_tokens'] = batch['text_tokens'].to(device)
        
        # Forward pass with mixed precision
        with amp.autocast('cuda'):
            codes, rvq_info = rvq.encode(batch['audio'])
            outputs = model(text_tokens=batch.get('text_tokens'), voice_codes=codes)
            
            targets = {
                'text_targets': batch.get('text_tokens'),
                'voice_targets': codes
            }
            loss = conversational_loss(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        
        # Print progress
        if (step + 1) % 10 == 0:
            avg_loss = sum(losses[-10:]) / len(losses[-10:])
            print(f"\nStep {step+1}: loss={avg_loss:.4f}")

    training_time = time.time() - start_time

    # Test inference speed
    print("\nTesting inference speed...")
    model.eval()
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            test_text = torch.randint(0, 1000, (1, 20)).to(device)
            _ = model(text_tokens=test_text)
    
    # Measure
    latencies = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            with amp.autocast('cuda'):
                test_text = torch.randint(0, 1000, (1, 20)).to(device)
                _ = model(text_tokens=test_text)
        
        torch.cuda.synchronize()
        latency = (time.time() - start) * 1000
        latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies)

    # Save model
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'model_state': model.state_dict(),
        'rvq_state': rvq.state_dict(),
        'config': config,
        'losses': losses,
        'training_time': training_time,
        'avg_latency_ms': avg_latency
    }
    
    checkpoint_path = checkpoint_dir / 'gpu_trained_model.pt'
    torch.save(checkpoint, checkpoint_path)

    # Summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Model parameters: {total_params/1e6:.1f}M")
    print(f"Training time: {training_time:.1f}s")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
    print(f"Average inference latency: {avg_latency:.1f}ms")
    print(f"GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"\nModel saved to: {checkpoint_path}")
    
    # Check if we achieved < 200ms
    if avg_latency < 200:
        print(f"\n✅ SUCCESS: Achieved {avg_latency:.1f}ms latency (< 200ms target!)")
    else:
        print(f"\n⚠️  Latency {avg_latency:.1f}ms is above 200ms target")
    
    return checkpoint_path, avg_latency

if __name__ == "__main__":
    checkpoint_path, latency = main()
    
    print("\nNext steps:")
    print("1. Run full benchmarks: python benchmark.py --checkpoint " + str(checkpoint_path))
    print("2. Deploy API: python serve_api.py --checkpoint " + str(checkpoint_path))
    print("3. Create demo: python interactive_demo.py --checkpoint " + str(checkpoint_path))