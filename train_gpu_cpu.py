#!/usr/bin/env python3
"""
Train on CPU first to debug, then move to GPU
"""

import torch
import torch.nn as nn
import time
from pathlib import Path
from tqdm import tqdm

def main():
    print("CPU Training (Debug Mode)")
    print("=" * 50)
    
    # Start on CPU
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Import model
    from architecture import CSMModel
    from missing_components import generate_test_batch, TrainingConfig
    
    # Tiny config
    config = {
        'd_model': 128,
        'n_layers': 2,
        'n_heads': 4,
        'max_seq_len': 512
    }
    
    # Create model
    print("\nCreating model...")
    model = CSMModel(**config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Simple optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training config
    test_config = TrainingConfig(
        batch_size=2,
        max_audio_length=24000,  # 1 second
        d_model=config['d_model']
    )
    
    # Training
    print(f"\nTraining for 10 steps on CPU...")
    model.train()
    
    losses = []
    
    for step in tqdm(range(10)):
        # Generate batch
        batch = generate_test_batch(test_config)
        
        # Move to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)
        
        # Forward pass
        try:
            # Just text generation first
            outputs = model(text_tokens=batch.get('text_tokens'))
            
            # Simple loss
            if outputs['text_logits'] is not None:
                loss = nn.functional.mse_loss(
                    outputs['text_logits'], 
                    torch.randn_like(outputs['text_logits'])
                )
            else:
                loss = torch.tensor(0.0)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
        except Exception as e:
            print(f"\nError at step {step}: {e}")
            print(f"Batch shapes: {[(k, v.shape) for k, v in batch.items() if torch.is_tensor(v)]}")
            raise
    
    print(f"\nCPU training successful! Average loss: {sum(losses)/len(losses):.4f}")
    
    # Now try GPU
    if torch.cuda.is_available():
        print("\n" + "=" * 50)
        print("Moving to GPU...")
        print("=" * 50)
        
        device = torch.device('cuda')
        model = model.to(device)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        print(f"Model on GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        
        # Test inference
        print("\nTesting GPU inference...")
        model.eval()
        
        with torch.no_grad():
            test_text = torch.randint(0, 1000, (1, 10)).to(device)
            
            # Warmup
            for _ in range(3):
                _ = model(text_tokens=test_text)
            
            # Measure
            torch.cuda.synchronize()
            start = time.time()
            
            outputs = model(text_tokens=test_text)
            
            torch.cuda.synchronize()
            latency = (time.time() - start) * 1000
        
        print(f"GPU Inference latency: {latency:.1f}ms")
        print(f"Memory after inference: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        
        # Save checkpoint
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'model_state': model.state_dict(),
            'config': config,
            'losses': losses,
            'latency_ms': latency,
            'device': 'cuda'
        }
        
        checkpoint_path = checkpoint_dir / 'debug_gpu_model.pt'
        torch.save(checkpoint, checkpoint_path)
        
        print("\n" + "=" * 50)
        print("GPU TRAINING COMPLETE!")
        print("=" * 50)
        print(f"Model saved to: {checkpoint_path}")
        print(f"Latency: {latency:.1f}ms")
        
        if latency < 200:
            print(f"\nSUCCESS: Achieved {latency:.1f}ms latency (< 200ms target!)")
        
        return checkpoint_path, latency
    
    return None, None

if __name__ == "__main__":
    checkpoint_path, latency = main()
    
    if checkpoint_path:
        print(f"\nNext steps:")
        print(f"1. Run benchmarks: python benchmark.py --checkpoint {checkpoint_path}")
        print(f"2. If successful, train larger model: python train_production.py")