import torch
import torch.nn.functional as F

# Test the problematic operation
batch_size = 2
time_steps = 3737  # From error message: 7353/2 ≈ 3737 (assuming batch size 2)
codebook_size = 1024
n_codebooks = 32

# Create test tensors
ema_cluster_size = torch.zeros(n_codebooks, codebook_size)
hard_indices = torch.randint(0, codebook_size, (batch_size, time_steps))

print(f"hard_indices shape: {hard_indices.shape}")

# Create one-hot encodings
encodings = F.one_hot(hard_indices, codebook_size).float()
print(f"encodings shape: {encodings.shape}")

# Sum over batch and time
encodings_sum = encodings.sum(dim=[0, 1])
print(f"encodings_sum shape: {encodings_sum.shape}")

# Test assignment
i = 0
ema_decay = 0.99

print(f"ema_cluster_size[i, :] shape: {ema_cluster_size[i, :].shape}")

# Try the assignment
try:
    ema_cluster_size[i, :] = ema_decay * ema_cluster_size[i, :] + (1 - ema_decay) * encodings_sum
    print("Assignment successful!")
except Exception as e:
    print(f"Error: {e}")
    
# Check if it's a device mismatch issue
print(f"ema_cluster_size device: {ema_cluster_size.device}")
print(f"encodings_sum device: {encodings_sum.device}")