"""
Analyze collected data and prepare for training
"""

import json
import wave
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

class DataAnalyzer:
    """Analyze collected conversational data"""
    
    def __init__(self, data_dir: str = "data/conversations"):
        self.data_dir = Path(data_dir)
        self.stats = defaultdict(list)
        
    def analyze_all(self):
        """Run all analyses"""
        print("=" * 60)
        print("CONVERSATIONAL DATA ANALYSIS")
        print("=" * 60)
        
        # Find all audio files
        audio_files = list(self.data_dir.rglob("*.wav"))
        print(f"\nFound {len(audio_files)} audio files")
        
        if not audio_files:
            print("No audio files found. Run collect_data.py first.")
            return
        
        # Analyze each file
        print("\nAnalyzing audio files...")
        for audio_file in audio_files[:100]:  # Analyze first 100
            self._analyze_file(audio_file)
        
        # Report statistics
        self._report_stats()
        
        # Generate training configs
        self._generate_configs()
        
        # Create data splits
        self._create_splits(audio_files)
    
    def _analyze_file(self, audio_path: Path) -> Dict:
        """Analyze single audio file"""
        try:
            with wave.open(str(audio_path), 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration = frames / rate
                
                self.stats['durations'].append(duration)
                self.stats['sample_rates'].append(rate)
                
                # Read audio data
                audio_data = wav.readframes(frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Simple voice activity
                energy = np.mean(np.abs(audio_array))
                silence_ratio = np.sum(np.abs(audio_array) < 500) / len(audio_array)
                
                self.stats['energy'].append(energy)
                self.stats['silence_ratio'].append(silence_ratio)
                
        except Exception as e:
            print(f"Error analyzing {audio_path}: {e}")
    
    def _report_stats(self):
        """Report data statistics"""
        print("\n" + "-"*40)
        print("DATA STATISTICS")
        print("-"*40)
        
        if self.stats['durations']:
            total_duration = sum(self.stats['durations'])
            avg_duration = np.mean(self.stats['durations'])
            
            print(f"Total duration: {total_duration/3600:.2f} hours")
            print(f"Average duration: {avg_duration:.2f} seconds")
            print(f"Duration range: {min(self.stats['durations']):.1f} - {max(self.stats['durations']):.1f}s")
            
            print(f"\nSample rates: {set(self.stats['sample_rates'])}")
            print(f"Average silence ratio: {np.mean(self.stats['silence_ratio']):.2%}")
            
            # Conversation length distribution
            print("\nDuration distribution:")
            bins = [0, 1, 2, 5, 10, 30, 60, float('inf')]
            hist = np.histogram(self.stats['durations'], bins)[0]
            
            for i, count in enumerate(hist):
                if i < len(bins) - 2:
                    print(f"  {bins[i]}-{bins[i+1]}s: {count} files")
                else:
                    print(f"  >{bins[i]}s: {count} files")
    
    def _generate_configs(self):
        """Generate optimal training configurations"""
        print("\n" + "-"*40)
        print("RECOMMENDED TRAINING CONFIGS")
        print("-"*40)
        
        configs = {
            "tiny_cpu": {
                "model": {
                    "d_model": 256,
                    "n_layers": 4,
                    "n_heads": 4,
                    "max_seq_len": 1024
                },
                "rvq": {
                    "n_codebooks": 8,
                    "codebook_size": 256,
                    "latent_dim": 128
                },
                "training": {
                    "batch_size": 1,
                    "learning_rate": 3e-4,
                    "gradient_accumulation": 4,
                    "mixed_precision": False
                },
                "estimated_time": "2-4 hours on CPU"
            },
            "small_gpu": {
                "model": {
                    "d_model": 768,
                    "n_layers": 12,
                    "n_heads": 12,
                    "max_seq_len": 2048
                },
                "rvq": {
                    "n_codebooks": 32,
                    "codebook_size": 1024,
                    "latent_dim": 512
                },
                "training": {
                    "batch_size": 8,
                    "learning_rate": 3e-4,
                    "gradient_accumulation": 2,
                    "mixed_precision": True
                },
                "estimated_time": "1-2 hours on RTX 3090"
            },
            "medium_gpu": {
                "model": {
                    "d_model": 1536,
                    "n_layers": 24,
                    "n_heads": 16,
                    "max_seq_len": 4096
                },
                "rvq": {
                    "n_codebooks": 32,
                    "codebook_size": 2048,
                    "latent_dim": 768
                },
                "training": {
                    "batch_size": 16,
                    "learning_rate": 2e-4,
                    "gradient_accumulation": 1,
                    "mixed_precision": True
                },
                "estimated_time": "4-6 hours on A100"
            }
        }
        
        # Save configs
        for name, config in configs.items():
            print(f"\n{name.upper()}:")
            print(f"  Parameters: ~{self._count_params(config):,}")
            print(f"  Memory required: ~{self._estimate_memory(config):.1f}GB")
            print(f"  Training time: {config['estimated_time']}")
            
            # Save config
            config_path = Path("configs") / f"{name}_config.json"
            config_path.parent.mkdir(exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  Saved to: {config_path}")
    
    def _count_params(self, config: dict) -> int:
        """Estimate parameter count"""
        d = config['model']['d_model']
        l = config['model']['n_layers']
        v = 128256  # Vocab size
        
        # Rough estimation
        embedding = v * d
        attention = l * (4 * d * d)  # Q,K,V,O projections
        ffn = l * (2 * d * d * 4)  # Two linear layers with 4x expansion
        rvq = config['rvq']['n_codebooks'] * config['rvq']['codebook_size'] * config['rvq']['latent_dim']
        
        return embedding + attention + ffn + rvq
    
    def _estimate_memory(self, config: dict) -> float:
        """Estimate GPU memory requirement"""
        params = self._count_params(config)
        
        # Rough estimation (params + gradients + optimizer states + activations)
        bytes_per_param = 4 if config['training']['mixed_precision'] else 8
        memory_multiplier = 20  # Conservative estimate
        
        return (params * bytes_per_param * memory_multiplier) / 1e9
    
    def _create_splits(self, audio_files: List[Path]):
        """Create train/val/test splits"""
        print("\n" + "-"*40)
        print("CREATING DATA SPLITS")
        print("-"*40)
        
        # Shuffle and split
        np.random.seed(42)
        indices = np.random.permutation(len(audio_files))
        
        n_train = int(0.8 * len(audio_files))
        n_val = int(0.1 * len(audio_files))
        
        train_files = [audio_files[i] for i in indices[:n_train]]
        val_files = [audio_files[i] for i in indices[n_train:n_train+n_val]]
        test_files = [audio_files[i] for i in indices[n_train+n_val:]]
        
        print(f"Train: {len(train_files)} files")
        print(f"Val: {len(val_files)} files")
        print(f"Test: {len(test_files)} files")
        
        # Save splits
        splits = {
            'train': [str(f.relative_to(self.data_dir)) for f in train_files],
            'val': [str(f.relative_to(self.data_dir)) for f in val_files],
            'test': [str(f.relative_to(self.data_dir)) for f in test_files]
        }
        
        with open(self.data_dir / "splits.json", 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"\nSplits saved to: {self.data_dir / 'splits.json'}")


def create_training_schedule():
    """Create optimal training schedule"""
    schedule = {
        "phase1_voice_understanding": {
            "duration": "2 epochs",
            "focus": "RVQ reconstruction",
            "loss_weights": {"rvq": 1.0, "text": 0.0, "cross_modal": 0.0}
        },
        "phase2_text_alignment": {
            "duration": "3 epochs", 
            "focus": "Text-voice alignment",
            "loss_weights": {"rvq": 0.5, "text": 0.5, "cross_modal": 0.5}
        },
        "phase3_conversation": {
            "duration": "5 epochs",
            "focus": "Full conversational training",
            "loss_weights": {"rvq": 0.3, "text": 0.3, "cross_modal": 0.4}
        }
    }
    
    print("\n" + "="*60)
    print("OPTIMAL TRAINING SCHEDULE")
    print("="*60)
    
    for phase, details in schedule.items():
        print(f"\n{phase.upper()}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    # Save schedule
    with open("training_schedule.json", 'w') as f:
        json.dump(schedule, f, indent=2)
    print("\nSchedule saved to: training_schedule.json")


if __name__ == "__main__":
    # Analyze data
    analyzer = DataAnalyzer()
    analyzer.analyze_all()
    
    # Create training schedule
    create_training_schedule()
    
    print("\n" + "="*60)
    print("READY FOR TRAINING!")
    print("="*60)
    print("\nNext steps:")
    print("1. Install CUDA PyTorch (run as admin):")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n2. Start training with generated config:")
    print("   python train.py --config configs/small_gpu_config.json")
    print("\n3. Monitor with tensorboard:")
    print("   tensorboard --logdir logs")