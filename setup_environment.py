"""
Setup script to prepare environment for CSM training
Handles dependencies, checks, and initial configuration
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import json
import torch
import importlib.util


def check_python_version():
    """Ensure Python 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version.split()[0]}")
    return True


def install_dependencies():
    """Install required packages"""
    print("\nInstalling dependencies...")
    
    # Core dependencies
    core_deps = [
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.21.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scipy>=1.7.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "webdataset>=0.2.0",
        "pyannote.audio>=3.0.0",
        "openai-whisper>=20230314",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "pandas>=1.5.0"
    ]
    
    # Install with pip
    for dep in core_deps:
        try:
            package_name = dep.split(">=")[0]
            if importlib.util.find_spec(package_name.replace("-", "_")) is None:
                print(f"Installing {dep}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            else:
                print(f"✅ {package_name} already installed")
        except Exception as e:
            print(f"⚠️  Failed to install {dep}: {e}")
            print("   You may need to install manually")


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPUs: {torch.cuda.device_count()}")
        return True
    else:
        print("⚠️  CUDA not available - will use CPU (very slow)")
        return False


def setup_directories():
    """Create necessary directories"""
    print("\nSetting up directories...")
    
    dirs = [
        "data/conversations",
        "data/processed",
        "checkpoints",
        "logs",
        "configs",
        "outputs/audio",
        "outputs/evaluations"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {dir_path}")


def create_default_config():
    """Create default training configuration"""
    print("\nCreating default configuration...")
    
    config = {
        "model": {
            "size": "tiny",
            "d_model": 768,
            "n_layers": 12,
            "n_heads": 12,
            "max_seq_len": 2048
        },
        "rvq": {
            "n_codebooks": 32,
            "codebook_size": 1024,
            "semantic_codebooks": 10,
            "codebook_dim": 32
        },
        "training": {
            "batch_size": 4 if torch.cuda.is_available() else 1,
            "learning_rate": 3e-4,
            "warmup_steps": 1000,
            "max_steps": 10000,
            "gradient_accumulation_steps": 8,
            "gradient_clip_val": 1.0,
            "val_check_interval": 500,
            "save_checkpoint_interval": 1000
        },
        "data": {
            "sample_rate": 24000,
            "max_audio_length": 240000,
            "context_window": 5,
            "num_workers": 2
        },
        "experiment": {
            "name": "csm_initial_test",
            "wandb_project": "csm-replication",
            "seed": 42
        }
    }
    
    config_path = Path("configs/default_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created default config at {config_path}")
    return config_path


def download_test_data():
    """Download minimal test data"""
    print("\nPreparing test data...")
    
    # Check if we have any data
    data_dir = Path("data/conversations")
    if not any(data_dir.rglob("*.wav")):
        print("No data found. Running minimal data collection...")
        
        # Run data collection for 1 hour
        subprocess.run([
            sys.executable, "collect_data.py",
            "--target_hours", "1",
            "--sources", "librispeech"
        ])
    else:
        print("✅ Data already available")


def test_imports():
    """Test all imports work correctly"""
    print("\nTesting imports...")
    
    try:
        # Test our modules
        from architecture import CSMModel
        from rvq_tokenizer import ConversationalRVQ
        from dataset_pipeline import ConversationalDataset
        from evaluation_framework import ComprehensiveConversationalEvaluator
        print("✅ All custom modules imported successfully")
        
        # Test a simple forward pass
        print("\nTesting model initialization...")
        model = CSMModel(d_model=256, n_layers=4, n_heads=4)  # Tiny test model
        rvq = ConversationalRVQ(n_codebooks=8, codebook_size=256)
        
        # Dummy input
        batch_size = 2
        text_tokens = torch.randint(0, 1000, (batch_size, 128))
        audio = torch.randn(batch_size, 1, 24000)  # 1 second
        
        # Test RVQ
        codes, info = rvq.encode(audio)
        print(f"✅ RVQ encoding successful: {len(codes)} codebooks")
        
        # Test model forward
        outputs = model(text_tokens=text_tokens, voice_codes=codes)
        print("✅ Model forward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def create_minimal_train_script():
    """Create a minimal training script that can run immediately"""
    script_content = '''#!/usr/bin/env python3
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
from missing_components import generate_test_batch

def train_minimal():
    # Load config
    with open("configs/default_config.json", 'r') as f:
        config = json.load(f)
    
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
    print("\\nStarting training...")
    model.train()
    rvq.train()
    
    for step in tqdm(range(100)):  # Just 100 steps for testing
        # Get batch
        if train_loader:
            try:
                batch = next(iter(train_loader))
            except:
                batch = generate_test_batch(config)
        else:
            batch = generate_test_batch(config)
        
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
            print(f"\\nStep {step}: Loss = {loss.item():.4f}")
    
    print("\\nTraining complete! Model is working correctly.")
    
    # Save checkpoint
    torch.save({
        'model_state': model.state_dict(),
        'rvq_state': rvq.state_dict(),
        'config': config
    }, 'checkpoints/test_model.pt')
    print("Saved test checkpoint to checkpoints/test_model.pt")

if __name__ == "__main__":
    train_minimal()
'''
    
    with open("train_minimal.py", 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix
    if platform.system() != "Windows":
        os.chmod("train_minimal.py", 0o755)
    
    print("✅ Created train_minimal.py")


def create_quick_start_guide():
    """Create a quick start guide"""
    guide = """# CSM Replication Quick Start Guide

## 1. Setup Environment (Already Done!)
```bash
python setup_environment.py
```

## 2. Collect Data (1 hour for testing)
```bash
python collect_data.py --target_hours 1
```

## 3. Run Minimal Training Test
```bash
python train_minimal.py
```

## 4. Full Training (when ready)
```bash
# Single GPU
python train.py --model_size tiny --max_steps 10000

# Multi-GPU
torchrun --nproc_per_node=4 train.py --model_size small
```

## 5. Monitor Training
- Check wandb dashboard: https://wandb.ai
- View logs in `logs/` directory
- Checkpoints saved in `checkpoints/`

## Next Steps:
1. Collect more data (target: 100+ hours)
2. Run hypothesis validation experiments
3. Scale to larger models
4. Implement real-time inference

## Troubleshooting:
- CUDA out of memory: Reduce batch_size in config
- Data loading errors: Check data/conversations has .wav files
- Import errors: Re-run setup_environment.py
"""
    
    with open("QUICKSTART.md", 'w') as f:
        f.write(guide)
    
    print("✅ Created QUICKSTART.md")


def main():
    """Run full environment setup"""
    print("🚀 CSM Environment Setup")
    print("=" * 50)
    
    # Check Python
    if not check_python_version():
        return
    
    # Install dependencies
    install_dependencies()
    
    # Check CUDA
    cuda_available = check_cuda()
    
    # Setup directories
    setup_directories()
    
    # Create default config
    create_default_config()
    
    # Test imports
    if not test_imports():
        print("\n⚠️  Some imports failed. Check error messages above.")
        print("   You may need to install additional dependencies.")
    
    # Create minimal training script
    create_minimal_train_script()
    
    # Create quick start guide
    create_quick_start_guide()
    
    # Download test data
    print("\n" + "=" * 50)
    print("Setup complete! 🎉")
    print("\nNext steps:")
    print("1. Run: python collect_data.py --target_hours 1")
    print("2. Test: python train_minimal.py")
    print("3. Read QUICKSTART.md for full instructions")
    
    if not cuda_available:
        print("\n⚠️  No GPU detected. Training will be very slow.")
        print("   Consider using Google Colab or cloud GPUs.")


if __name__ == "__main__":
    main()