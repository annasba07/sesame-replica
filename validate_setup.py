"""
Simple validation script without unicode issues
"""

import sys
import importlib
import torch
import subprocess
from pathlib import Path


def check_imports():
    """Check all required imports"""
    print("Checking imports...")
    
    required_modules = [
        'torch',
        'torchaudio',
        'numpy',
        'librosa',
        'transformers',
        'pytorch_lightning',
        'wandb'
    ]
    
    missing = []
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  [OK] {module}")
        except ImportError:
            print(f"  [MISSING] {module}")
            missing.append(module)
    
    return len(missing) == 0


def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU...")
    if torch.cuda.is_available():
        print(f"  [OK] GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  [OK] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("  [WARNING] No GPU detected - training will be slow")
        return False


def test_models():
    """Test model imports and basic functionality"""
    print("\nTesting model imports...")
    
    try:
        from architecture import CSMModel
        from rvq_tokenizer import ConversationalRVQ
        from dataset_pipeline import ConversationalDataset
        
        # Create tiny models
        model = CSMModel(d_model=256, n_layers=4, n_heads=4)
        rvq = ConversationalRVQ(n_codebooks=8, codebook_size=256)
        
        # Test forward pass
        batch_size = 2
        text_tokens = torch.randint(0, 1000, (batch_size, 128))
        audio = torch.randn(batch_size, 1, 24000)
        
        codes, _ = rvq.encode(audio)
        outputs = model(text_tokens=text_tokens, voice_codes=codes)
        
        print("  [OK] Models working correctly")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Model test failed: {e}")
        return False


def check_data():
    """Check if we have any data"""
    print("\nChecking data...")
    data_dir = Path("data/conversations")
    
    if data_dir.exists():
        wav_files = list(data_dir.rglob("*.wav"))
        if wav_files:
            print(f"  [OK] Found {len(wav_files)} audio files")
            return True
        else:
            print("  [INFO] No audio files found yet")
            return False
    else:
        print("  [INFO] Data directory doesn't exist")
        return False


def main():
    print("CSM Validation Check")
    print("=" * 50)
    
    results = {
        'imports': check_imports(),
        'gpu': check_gpu(),
        'models': test_models(),
        'data': check_data()
    }
    
    print("\nSummary:")
    print("-" * 30)
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL" if check != 'gpu' else "WARN"
        print(f"{check:10} : {status}")
    
    if results['imports'] and results['models']:
        print("\nSystem ready for training!")
        print("\nNext steps:")
        print("1. Collect data: python collect_data.py --target_hours 10")
        print("2. Train model: python train_minimal.py")
    else:
        print("\nPlease fix issues before proceeding")


if __name__ == "__main__":
    main()