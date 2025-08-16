"""Minimal test to check if models can be created and run"""

import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Test 1: Create tiny RVQ
print("\n1. Creating tiny RVQ...")
from rvq_tokenizer import ConversationalRVQ

try:
    rvq = ConversationalRVQ(n_codebooks=2, codebook_size=64, latent_dim=64, channels=[16, 32, 64])
    print("RVQ created successfully")
    
    # Test encode
    audio = torch.randn(1, 1, 2400)  # 0.1 second
    codes, info = rvq.encode(audio)
    print(f"Encoded to {len(codes)} codes")
except Exception as e:
    print(f"RVQ failed: {e}")

# Test 2: Create tiny CSM
print("\n2. Creating tiny CSM...")
from architecture import CSMModel

try:
    model = CSMModel(d_model=32, n_layers=1, n_heads=1, rvq_codebooks=2, rvq_codebook_size=64)
    print("CSM created successfully")
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
except Exception as e:
    print(f"CSM failed: {e}")

# Test 3: Forward pass
print("\n3. Testing forward pass...")
try:
    text = torch.randint(0, 100, (1, 5))
    outputs = model(text_tokens=text, voice_codes=codes)
    print("Forward pass successful")
    print(f"Output keys: {list(outputs.keys())}")
except Exception as e:
    print(f"Forward pass failed: {e}")

print("\nBasic tests complete!")