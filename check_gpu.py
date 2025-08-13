"""Check GPU availability and provide installation instructions"""

import torch
import platform

print("System Information")
print("=" * 50)
print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("\nCUDA not available. To install PyTorch with CUDA:")
    print("\n1. Open a new Command Prompt as Administrator")
    print("2. Run this command:")
    print("\npip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade --force-reinstall")
    print("\n3. If that doesn't work, try:")
    print("pip uninstall torch torchvision torchaudio -y")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
print("\nNote: You may need to restart your terminal/IDE after installation.")