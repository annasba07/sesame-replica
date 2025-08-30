"""Test the quantizer loop"""
import torch
import torch.nn.functional as F
from rvq_tokenizer import ConversationalRVQ

# Create small RVQ for testing
device = torch.device('cpu')
rvq = ConversationalRVQ(n_codebooks=4, codebook_size=256, semantic_codebooks=2).to(device)

# Create test data
batch_size = 1
time_steps = 100
audio = torch.randn(batch_size, 1, time_steps * 32)  # 32 is downsampling factor

# Encode
latent, (semantic, acoustic) = rvq.encoder(audio)
print(f"Latent shape: {latent.shape}")

# Create semantic mask
semantic_energy = semantic.abs().mean(dim=1, keepdim=True)
semantic_mask = (semantic_energy > semantic_energy.mean()).float()

# Project input
x = rvq.quantizer.input_proj(latent)
x = x.transpose(1, 2)
print(f"x shape after transpose: {x.shape}")

# Initialize variables
residual = x
quantized = torch.zeros_like(x)
indices = []

# Test first iteration
i = 0
codebook = rvq.quantizer.codebooks[i]
print(f"\nIteration {i}:")
print(f"Residual shape: {residual.shape}")
print(f"Codebook shape: {codebook.shape}")

# Apply mask
if i < rvq.quantizer.semantic_codebooks and semantic_mask is not None:
    print(f"Applying semantic mask...")
    residual_masked = residual * semantic_mask.transpose(1, 2)
    print(f"Residual after mask shape: {residual_masked.shape}")

# Compute distances
distances = torch.cdist(residual, codebook)
print(f"Distances shape: {distances.shape}")

# Get hard indices
hard_indices = distances.argmin(dim=-1)
print(f"Hard indices shape: {hard_indices.shape}")

# Create one-hot
encodings = F.one_hot(hard_indices, rvq.quantizer.codebook_size).float()
print(f"Encodings shape: {encodings.shape}")

# Sum encodings
encodings_sum = encodings.sum(dim=[0, 1])
print(f"Encodings sum shape: {encodings_sum.shape}")

# Test EMA update
print(f"\nEMA shapes:")
print(f"ema_cluster_size shape: {rvq.quantizer.ema_cluster_size.shape}")
print(f"ema_cluster_size[i] shape: {rvq.quantizer.ema_cluster_size[i].shape}")

# Try the assignment
try:
    rvq.quantizer.ema_cluster_size[i] = (
        0.99 * rvq.quantizer.ema_cluster_size[i] +
        0.01 * encodings_sum
    )
    print("EMA update successful!")
except Exception as e:
    print(f"EMA update error: {e}")
    print(f"Error type: {type(e)}")