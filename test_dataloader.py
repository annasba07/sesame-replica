"""Test what the dataloader returns"""
from dataset_pipeline import create_dataloader

# Create dataloader
loader = create_dataloader(["data/conversations"], batch_size=1, num_workers=0)

# Get one batch
batch = next(iter(loader))

print("Batch keys:", batch.keys())
print("Audio shape:", batch['audio'].shape)
if 'text_tokens' in batch:
    print("Text tokens shape:", batch['text_tokens'].shape)

# Test with RVQ
from rvq_tokenizer import ConversationalRVQ
import torch

device = torch.device('cpu')
rvq = ConversationalRVQ(n_codebooks=8, codebook_size=256).to(device)

# Test encode
print("\nTesting RVQ encode...")
audio = batch['audio'].to(device)
print(f"Input audio shape: {audio.shape}")

# Get the encoder output
latent, (semantic, acoustic) = rvq.encoder(audio)
print(f"Latent shape after encoder: {latent.shape}")
print(f"Semantic shape: {semantic.shape}")
print(f"Acoustic shape: {acoustic.shape}")

# Test quantizer directly
print("\nTesting quantizer...")
semantic_energy = semantic.abs().mean(dim=1, keepdim=True)
semantic_mask = (semantic_energy > semantic_energy.mean()).float()
print(f"Semantic mask shape: {semantic_mask.shape}")

# Debug the issue
x = rvq.quantizer.input_proj(latent)
print(f"After input_proj: {x.shape}")
x_transposed = x.transpose(1, 2)
print(f"After transpose: {x_transposed.shape}")