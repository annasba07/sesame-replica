#!/usr/bin/env python3
"""Debug version of training script"""

import torch
import json
from pathlib import Path
from rvq_tokenizer import ConversationalRVQ
from dataset_pipeline import create_dataloader

# Load config
config = {
    'model': {'d_model': 768, 'n_layers': 12, 'n_heads': 12},
    'rvq': {'n_codebooks': 32, 'codebook_size': 1024},
    'training': {'batch_size': 1, 'learning_rate': 3e-4}
}

# Setup device
device = torch.device('cpu')
print(f"Using device: {device}")

# Create RVQ only
print(f"Creating RVQ with codebook_size={config['rvq']['codebook_size']}")
rvq = ConversationalRVQ(
    n_codebooks=config['rvq']['n_codebooks'],
    codebook_size=config['rvq']['codebook_size']
).to(device)

# Check quantizer parameters
print(f"Quantizer codebook_size: {rvq.quantizer.codebook_size}")
print(f"Quantizer n_codebooks: {rvq.quantizer.n_codebooks}")
print(f"First codebook shape: {rvq.quantizer.codebooks[0].shape}")

# Load data
try:
    train_loader = create_dataloader(
        ["data/conversations"],
        batch_size=1,
        num_workers=0
    )
    print("Using real data")
    
    # Get one batch
    batch = next(iter(train_loader))
    print(f"Audio batch shape: {batch['audio'].shape}")
    
    # Try encoding
    print("\nTrying to encode...")
    codes, rvq_info = rvq.encode(batch['audio'])
    print("Encoding successful!")
    
except Exception as e:
    print(f"Error during encoding: {e}")
    import traceback
    traceback.print_exc()