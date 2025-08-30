import torch

# Test cdist behavior
batch_size = 1
time_steps = 10
codebook_dim = 32
codebook_size = 1024

# Create test tensors
residual = torch.randn(batch_size, time_steps, codebook_dim)
codebook = torch.randn(codebook_size, codebook_dim)

print(f"residual shape: {residual.shape}")
print(f"codebook shape: {codebook.shape}")

# Test 1: Direct cdist
try:
    distances1 = torch.cdist(residual, codebook)
    print(f"Direct cdist shape: {distances1.shape}")
except Exception as e:
    print(f"Direct cdist error: {e}")

# Test 2: With unsqueeze
try:
    distances2 = torch.cdist(residual, codebook.unsqueeze(0))
    print(f"Unsqueeze cdist shape: {distances2.shape}")
except Exception as e:
    print(f"Unsqueeze cdist error: {e}")

# Test 3: Manual distance computation
# Expand dimensions for broadcasting
residual_exp = residual.unsqueeze(2)  # [B, T, 1, D]
codebook_exp = codebook.unsqueeze(0).unsqueeze(0)  # [1, 1, K, D]
distances3 = ((residual_exp - codebook_exp) ** 2).sum(dim=-1).sqrt()
print(f"Manual computation shape: {distances3.shape}")

# Test argmin
if 'distances1' in locals():
    indices = distances1.argmin(dim=-1)
    print(f"Indices shape: {indices.shape}")
elif 'distances3' in locals():
    indices = distances3.argmin(dim=-1)
    print(f"Indices shape: {indices.shape}")
    
# Test one-hot
one_hot = torch.nn.functional.one_hot(indices, codebook_size)
print(f"One-hot shape: {one_hot.shape}")
print(f"One-hot sum over [0,1] shape: {one_hot.sum(dim=[0,1]).shape}")