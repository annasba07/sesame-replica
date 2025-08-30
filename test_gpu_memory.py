#!/usr/bin/env python3
"""
Test GPU memory usage step by step
"""

import torch
import torch.nn as nn

print("Testing GPU Memory")
print("=" * 50)

device = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Free Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")

# Clear everything
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("\n1. Creating tiny test tensor...")
try:
    x = torch.randn(1, 1, 1000).to(device)
    print(f"   Success! Memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
except Exception as e:
    print(f"   Failed: {e}")

print("\n2. Creating small conv layer...")
try:
    conv = nn.Conv1d(1, 16, 3).to(device)
    print(f"   Success! Memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
except Exception as e:
    print(f"   Failed: {e}")

print("\n3. Running forward pass...")
try:
    with torch.no_grad():
        y = conv(x)
    print(f"   Success! Output shape: {y.shape}")
    print(f"   Memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
except Exception as e:
    print(f"   Failed: {e}")

print("\n4. Testing batch processing...")
try:
    batch = torch.randn(2, 1, 24000).to(device)
    print(f"   Created batch. Memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
except Exception as e:
    print(f"   Failed: {e}")

print("\n5. Testing RVQ components...")
try:
    # Minimal RVQ encoder
    encoder = nn.Sequential(
        nn.Conv1d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv1d(32, 64, 3, padding=1)
    ).to(device)
    
    with torch.no_grad():
        encoded = encoder(batch)
    
    print(f"   Success! Encoded shape: {encoded.shape}")
    print(f"   Memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
except Exception as e:
    print(f"   Failed: {e}")

print("\n6. Testing with amp autocast...")
try:
    import torch.amp as amp
    
    with amp.autocast('cuda'):
        with torch.no_grad():
            encoded = encoder(batch)
    
    print(f"   Success with autocast!")
    print(f"   Memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
except Exception as e:
    print(f"   Failed: {e}")

print("\nMemory Summary:")
print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
print(f"Current memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

# Check if other processes are using GPU
print("\n7. Checking for other GPU processes...")
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.stdout.strip():
        print("Other processes using GPU:")
        print(result.stdout)
    else:
        print("No other processes using GPU")
except Exception as e:
    print(f"Could not check processes: {e}")