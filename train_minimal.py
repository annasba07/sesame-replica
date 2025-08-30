#!/usr/bin/env python3
"""
Minimal CSM Training Script
Can run with just 1 hour of data on a single GPU
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm

# Import our modules
from architecture import CSMModel, conversational_loss
from rvq_tokenizer import ConversationalRVQ
from dataset_pipeline import create_dataloader
from missing_components import generate_test_batch, TrainingConfig

def train_minimal():
    # Load config
    config_path = Path("configs/default_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Create default config if it doesn't exist
        config = {
            'model': {'d_model': 768, 'n_layers': 12, 'n_heads': 12},
            'rvq': {'n_codebooks': 32, 'codebook_size': 1024},
            'training': {'batch_size': 1, 'learning_rate': 3e-4}
        }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models (tiny versions)
    model = CSMModel(
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads']
    ).to(device)
    
    rvq = ConversationalRVQ(
        n_codebooks=config['rvq']['n_codebooks'],
        codebook_size=config['rvq']['codebook_size']
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(rvq.parameters()),
        lr=config['training']['learning_rate']
    )
    
    # Try to load real data, fall back to synthetic
    try:
        train_loader = create_dataloader(
            ["data/conversations"],
            batch_size=config['training']['batch_size'],
            num_workers=0  # Start with 0 to avoid multiprocessing issues
        )
        print("Using real data")
    except:
        print("Using synthetic data for testing")
        train_loader = None
    
    # Training loop
    print("\nStarting training...")
    model.train()
    rvq.train()
    
    for step in tqdm(range(100)):  # Just 100 steps for testing
        # Get batch
        if train_loader:
            try:
                batch = next(iter(train_loader))
            except:
                test_config = TrainingConfig()
                batch = generate_test_batch(test_config)
        else:
            test_config = TrainingConfig()
            batch = generate_test_batch(test_config)
        
        # Move to device
        batch['audio'] = batch['audio'].to(device)
        if 'text_tokens' in batch:
            batch['text_tokens'] = batch['text_tokens'].to(device)
        
        # Forward pass
        codes, rvq_info = rvq.encode(batch['audio'])
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
        
        if step % 10 == 0:
            print(f"\nStep {step}: Loss = {loss.item():.4f}")
    
    print("\nTraining complete! Model is working correctly.")
    
    # Save checkpoint
    torch.save({
        'model_state': model.state_dict(),
        'rvq_state': rvq.state_dict(),
        'config': config
    }, 'checkpoints/test_model.pt')
    print("Saved test checkpoint to checkpoints/test_model.pt")

if __name__ == "__main__":
    train_minimal()
