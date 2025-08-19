#!/usr/bin/env python3
"""Simple training script - 5 steps only"""

import torch
import time
from pathlib import Path

# Import our modules
from architecture import CSMModel, conversational_loss
from rvq_tokenizer import ConversationalRVQ
from missing_components import generate_test_batch, TrainingConfig

print("Simple CSM Training Test")
print("=" * 50)

# Config - ultra tiny for speed
config = {
    'model': {'d_model': 64, 'n_layers': 2, 'n_heads': 2},
    'rvq': {'n_codebooks': 4, 'codebook_size': 128},
    'training': {'batch_size': 1, 'learning_rate': 1e-3}
}

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Create tiny models
model = CSMModel(
    d_model=config['model']['d_model'],
    n_layers=config['model']['n_layers'],
    n_heads=config['model']['n_heads']
).to(device)

rvq = ConversationalRVQ(
    n_codebooks=config['rvq']['n_codebooks'],
    codebook_size=config['rvq']['codebook_size']
).to(device)

# Count parameters
model_params = sum(p.numel() for p in model.parameters())
rvq_params = sum(p.numel() for p in rvq.parameters())
print(f"Model parameters: {model_params:,}")
print(f"RVQ parameters: {rvq_params:,}")
print(f"Total parameters: {model_params + rvq_params:,}")

# Optimizer
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(rvq.parameters()),
    lr=config['training']['learning_rate']
)

# Train for 5 steps
print("\nTraining for 5 steps...")
model.train()
rvq.train()

test_config = TrainingConfig()
losses = []
times = []

for step in range(5):
    start = time.time()
    
    # Generate batch
    batch = generate_test_batch(test_config)
    batch['audio'] = batch['audio'].to(device)
    if 'text_tokens' in batch:
        batch['text_tokens'] = batch['text_tokens'].to(device)
    
    # Forward
    codes, _ = rvq.encode(batch['audio'])
    outputs = model(text_tokens=batch.get('text_tokens'), voice_codes=codes)
    
    # Loss
    targets = {
        'text_targets': batch.get('text_tokens'),
        'voice_targets': codes
    }
    loss = conversational_loss(outputs, targets)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record
    step_time = time.time() - start
    losses.append(loss.item())
    times.append(step_time)
    
    print(f"Step {step+1}: loss={loss.item():.4f}, time={step_time:.2f}s")

# Summary
print("\n" + "=" * 50)
print("TRAINING SUMMARY")
print("=" * 50)
print(f"Average time per step: {sum(times)/len(times):.2f}s")
print(f"Initial loss: {losses[0]:.4f}")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Loss reduction: {((losses[0] - losses[-1])/losses[0]*100):.1f}%")

# Save
Path("checkpoints").mkdir(exist_ok=True)
torch.save({
    'model_state': model.state_dict(),
    'rvq_state': rvq.state_dict(),
    'config': config,
    'losses': losses
}, 'checkpoints/simple_training.pt')
print("\nCheckpoint saved to checkpoints/simple_training.pt")

# Test generation
print("\nTesting generation...")
model.eval()
rvq.eval()

with torch.no_grad():
    # Generate from text
    test_text = torch.randint(0, 1000, (1, 5)).to(device)
    outputs = model(text_tokens=test_text)
    print(f"Generated voice logits shape: {outputs['voice_logits'].shape}")

print("\nTraining test complete!")