#!/usr/bin/env python3
"""Quick training test - just 10 steps"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
import time

# Import our modules
from architecture import CSMModel, conversational_loss
from rvq_tokenizer import ConversationalRVQ
from dataset_pipeline import create_dataloader
from missing_components import generate_test_batch, TrainingConfig

def train_quick():
    # Create checkpoint dir
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Config
    config = {
        'model': {'d_model': 256, 'n_layers': 4, 'n_heads': 4},  # Tiny model
        'rvq': {'n_codebooks': 8, 'codebook_size': 256},  # Small codebooks
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
    
    # Use synthetic data for speed
    test_config = TrainingConfig()
    
    # Training loop
    print("\nStarting quick training test...")
    model.train()
    rvq.train()
    
    losses = []
    start_time = time.time()
    
    for step in tqdm(range(10)):  # Just 10 steps
        # Get batch
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
        
        losses.append(loss.item())
        
        if step % 2 == 0:
            print(f"\nStep {step}: Loss = {loss.item():.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f} seconds")
    print(f"Average loss: {sum(losses)/len(losses):.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    
    # Save checkpoint
    torch.save({
        'model_state': model.state_dict(),
        'rvq_state': rvq.state_dict(),
        'config': config,
        'losses': losses
    }, 'checkpoints/quick_test.pt')
    print("\nSaved checkpoint to checkpoints/quick_test.pt")
    
    return losses[-1] < losses[0]  # Check if loss decreased

if __name__ == "__main__":
    success = train_quick()
    if success:
        print("\n✅ Training test PASSED - loss decreased")
    else:
        print("\n❌ Training test FAILED - loss did not decrease")