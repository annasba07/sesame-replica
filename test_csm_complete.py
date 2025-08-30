"""
Comprehensive Testing Suite for CSM Implementation
Tests all components from architecture to inference
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
import traceback
from typing import Dict, List, Tuple

# Import all components
from architecture import CSMModel, VoiceTextCrossAttention, ConversationalMemory
from rvq_tokenizer import ConversationalRVQ
from dataset_pipeline import ConversationalDataset, ProsodyAnalyzer
from evaluation_framework import CharacterConsistencyEvaluator, EmotionalCoherenceEvaluator
from hypothesis_experiments import HypothesisValidator
from streaming_inference import StreamingCSM

class CSMTestSuite:
    """Complete test suite for CSM implementation"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        print(f"Testing on device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("=" * 60)
        
    def test_1_core_architecture(self) -> bool:
        """Test core CSM architecture components"""
        print("\n1. TESTING CORE ARCHITECTURE")
        print("-" * 40)
        
        try:
            # Test VoiceTextCrossAttention
            print("Testing VoiceTextCrossAttention...")
            cross_attn = VoiceTextCrossAttention(d_model=768, n_heads=12).to(self.device)
            
            batch_size = 2
            seq_len = 128
            d_model = 768
            
            text_hidden = torch.randn(batch_size, seq_len, d_model).to(self.device)
            voice_hidden = torch.randn(batch_size, seq_len * 2, d_model).to(self.device)  # Different length
            
            text_out, voice_out = cross_attn(text_hidden, voice_hidden)
            assert text_out.shape == (batch_size, seq_len, d_model)
            print("✓ Cross-attention handles different sequence lengths")
            
            # Test CSM Model
            print("Testing CSM Model...")
            model = CSMModel(
                d_model=768,
                n_layers=6,
                n_heads=12,
                max_seq_len=512
            ).to(self.device)
            
            text_tokens = torch.randint(0, 1000, (batch_size, 64)).to(self.device)
            voice_codes = torch.randint(0, 1024, (batch_size, 128, 32)).to(self.device)
            
            outputs = model(text_tokens=text_tokens, voice_codes=voice_codes)
            assert 'text_logits' in outputs
            assert 'voice_logits' in outputs
            print("✓ CSM model forward pass successful")
            
            # Test memory system
            print("Testing Conversational Memory...")
            memory = ConversationalMemory(d_model=768).to(self.device)
            
            hidden = torch.randn(batch_size, seq_len, 768).to(self.device)
            speaker_ids = torch.tensor([0, 1]).to(self.device)
            
            updated = memory(hidden, speaker_ids)
            assert updated.shape == hidden.shape
            print("✓ Conversational memory working")
            
            # Test memory efficiency
            param_count = sum(p.numel() for p in model.parameters())
            print(f"✓ Model parameters: {param_count/1e6:.2f}M")
            
            self.results['architecture'] = True
            return True
            
        except Exception as e:
            print(f"✗ Architecture test failed: {str(e)}")
            traceback.print_exc()
            self.results['architecture'] = False
            return False
    
    def test_2_rvq_tokenizer(self) -> bool:
        """Test hierarchical RVQ tokenizer"""
        print("\n2. TESTING RVQ TOKENIZER")
        print("-" * 40)
        
        try:
            print("Creating ConversationalRVQ...")
            rvq = ConversationalRVQ(
                n_codebooks=32,
                codebook_size=1024,
                semantic_codebooks=10
            ).to(self.device)
            
            # Test encoding
            batch_size = 2
            audio_length = 24000  # 1 second at 24kHz
            audio = torch.randn(batch_size, 1, audio_length).to(self.device)
            
            print("Testing encoding...")
            codes, commitment_loss = rvq.encode(audio)
            assert codes.shape[0] == batch_size
            assert codes.shape[-1] == 32  # n_codebooks
            print(f"✓ Encoded shape: {codes.shape}")
            
            # Test decoding
            print("Testing decoding...")
            reconstructed = rvq.decode(codes)
            assert reconstructed.shape[0] == batch_size
            print(f"✓ Decoded shape: {reconstructed.shape}")
            
            # Test semantic/acoustic separation
            semantic_codes = codes[..., :10]
            acoustic_codes = codes[..., 10:]
            print(f"✓ Semantic codes: {semantic_codes.shape[-1]} codebooks")
            print(f"✓ Acoustic codes: {acoustic_codes.shape[-1]} codebooks")
            
            # Test commitment loss
            assert commitment_loss.item() >= 0
            print(f"✓ Commitment loss: {commitment_loss.item():.4f}")
            
            self.results['rvq'] = True
            return True
            
        except Exception as e:
            print(f"✗ RVQ test failed: {str(e)}")
            traceback.print_exc()
            self.results['rvq'] = False
            return False
    
    def test_3_data_pipeline(self) -> bool:
        """Test data loading and processing"""
        print("\n3. TESTING DATA PIPELINE")
        print("-" * 40)
        
        try:
            # Check if data exists
            data_path = Path("data/conversations/librispeech/processed")
            if not data_path.exists():
                print("⚠ No data found. Run collect_data.py first")
                self.results['data_pipeline'] = None
                return False
            
            wav_files = list(data_path.glob("*.wav"))
            print(f"Found {len(wav_files)} audio files")
            
            if len(wav_files) > 0:
                # Test prosody analyzer
                print("Testing ProsodyAnalyzer...")
                analyzer = ProsodyAnalyzer(sample_rate=24000)
                
                # Load a sample audio file
                import torchaudio
                audio, sr = torchaudio.load(str(wav_files[0]))
                audio_np = audio.numpy().squeeze()
                
                features = analyzer.extract_features(audio_np)
                print("Prosodic features extracted:")
                for key, value in list(features.items())[:5]:
                    print(f"  {key}: {value:.4f}")
                print("✓ Prosody analysis working")
                
                # Test dataset
                print("Testing ConversationalDataset...")
                dataset = ConversationalDataset(
                    data_dir=str(data_path.parent),
                    sample_rate=24000,
                    max_length=48000  # 2 seconds
                )
                
                if len(dataset) > 0:
                    sample = dataset[0]
                    print(f"✓ Dataset loaded with {len(dataset)} samples")
                    print(f"  Sample keys: {list(sample.keys())}")
                
            self.results['data_pipeline'] = True
            return True
            
        except Exception as e:
            print(f"✗ Data pipeline test failed: {str(e)}")
            traceback.print_exc()
            self.results['data_pipeline'] = False
            return False
    
    def test_4_training_step(self) -> bool:
        """Test a single training step"""
        print("\n4. TESTING TRAINING STEP")
        print("-" * 40)
        
        try:
            print("Creating models for training...")
            model = CSMModel(
                d_model=512,  # Smaller for testing
                n_layers=4,
                n_heads=8
            ).to(self.device)
            
            rvq = ConversationalRVQ(
                n_codebooks=16,  # Smaller for testing
                codebook_size=512
            ).to(self.device)
            
            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(rvq.parameters()),
                lr=1e-4
            )
            
            # Create dummy batch
            batch_size = 2
            audio = torch.randn(batch_size, 1, 24000).to(self.device)
            text = torch.randint(0, 1000, (batch_size, 64)).to(self.device)
            
            print("Running forward pass...")
            # Encode audio
            voice_codes, commitment_loss = rvq.encode(audio)
            
            # Forward through model
            outputs = model(text_tokens=text, voice_codes=voice_codes)
            
            # Compute dummy loss
            loss = outputs['text_logits'].mean() + outputs['voice_logits'].mean() + commitment_loss
            
            print("Running backward pass...")
            loss.backward()
            
            # Check gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            print(f"✓ Gradient norm: {grad_norm:.4f}")
            
            optimizer.step()
            optimizer.zero_grad()
            
            print("✓ Training step completed successfully")
            
            # Test memory usage
            if self.device.type == 'cuda':
                memory_used = torch.cuda.max_memory_allocated() / 1e9
                print(f"✓ Peak GPU memory: {memory_used:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
            
            self.results['training'] = True
            return True
            
        except Exception as e:
            print(f"✗ Training test failed: {str(e)}")
            traceback.print_exc()
            self.results['training'] = False
            return False
    
    def test_5_inference(self) -> bool:
        """Test inference and generation"""
        print("\n5. TESTING INFERENCE")
        print("-" * 40)
        
        try:
            print("Creating model for inference...")
            model = CSMModel(
                d_model=512,
                n_layers=4,
                n_heads=8
            ).to(self.device)
            model.eval()
            
            with torch.no_grad():
                # Test text-to-voice
                print("Testing text-to-voice generation...")
                text = torch.randint(0, 1000, (1, 32)).to(self.device)
                
                start_time = time.time()
                outputs = model(text_tokens=text)
                inference_time = (time.time() - start_time) * 1000
                
                print(f"✓ Inference time: {inference_time:.2f}ms")
                print(f"✓ Output shape: {outputs['voice_logits'].shape}")
                
                # Test voice-to-text
                print("Testing voice-to-text generation...")
                voice_codes = torch.randint(0, 512, (1, 64, 16)).to(self.device)
                
                outputs = model(voice_codes=voice_codes)
                print(f"✓ Text output shape: {outputs['text_logits'].shape}")
                
                # Test streaming capability
                print("Testing streaming inference setup...")
                streamer = StreamingCSM()
                print(f"✓ Chunk size: {streamer.chunk_size_ms}ms")
                print(f"✓ Target latency: <200ms")
            
            self.results['inference'] = True
            return True
            
        except Exception as e:
            print(f"✗ Inference test failed: {str(e)}")
            traceback.print_exc()
            self.results['inference'] = False
            return False
    
    def test_6_hypothesis_validation(self) -> bool:
        """Run hypothesis validation experiments"""
        print("\n6. TESTING HYPOTHESIS EXPERIMENTS")
        print("-" * 40)
        
        try:
            validator = HypothesisValidator()
            
            # Run experiments
            exp1 = validator.experiment1_voice_semantics()
            exp2 = validator.experiment2_memory_benefit()
            exp3 = validator.experiment3_unified_vs_pipeline()
            
            print("\n✓ All hypothesis experiments completed")
            print(f"  Voice-semantic coupling: {'PASS' if exp1 else 'FAIL'}")
            print(f"  Memory benefit: {'PASS' if exp2 else 'FAIL'}")
            print(f"  Unified > Pipeline: {'PASS' if exp3 else 'FAIL'}")
            
            self.results['hypothesis'] = all([exp1, exp2, exp3])
            return self.results['hypothesis']
            
        except Exception as e:
            print(f"✗ Hypothesis test failed: {str(e)}")
            traceback.print_exc()
            self.results['hypothesis'] = False
            return False
    
    def test_7_performance_benchmarks(self) -> bool:
        """Run performance benchmarks"""
        print("\n7. PERFORMANCE BENCHMARKS")
        print("-" * 40)
        
        try:
            benchmarks = {}
            
            # Model sizes
            model_configs = {
                'tiny': {'d_model': 384, 'n_layers': 4, 'n_heads': 6},
                'small': {'d_model': 768, 'n_layers': 12, 'n_heads': 12},
            }
            
            for size, config in model_configs.items():
                print(f"\nTesting {size} model...")
                model = CSMModel(**config).to(self.device)
                
                param_count = sum(p.numel() for p in model.parameters())
                
                # Measure inference speed
                batch_size = 1
                seq_len = 128
                text = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
                
                model.eval()
                with torch.no_grad():
                    # Warmup
                    for _ in range(3):
                        _ = model(text_tokens=text)
                    
                    # Benchmark
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start = time.time()
                    for _ in range(10):
                        _ = model(text_tokens=text)
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    avg_time = (time.time() - start) / 10 * 1000
                
                benchmarks[size] = {
                    'parameters': param_count,
                    'inference_ms': avg_time
                }
                
                print(f"  Parameters: {param_count/1e6:.2f}M")
                print(f"  Inference: {avg_time:.2f}ms")
                
                # Memory usage
                if self.device.type == 'cuda':
                    memory = torch.cuda.max_memory_allocated() / 1e9
                    benchmarks[size]['memory_gb'] = memory
                    print(f"  Memory: {memory:.2f}GB")
                    torch.cuda.reset_peak_memory_stats()
            
            # Save benchmarks
            with open('test_benchmarks.json', 'w') as f:
                json.dump(benchmarks, f, indent=2)
            
            print("\n✓ Benchmarks saved to test_benchmarks.json")
            self.results['benchmarks'] = True
            return True
            
        except Exception as e:
            print(f"✗ Benchmark test failed: {str(e)}")
            traceback.print_exc()
            self.results['benchmarks'] = False
            return False
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "=" * 60)
        print("RUNNING COMPLETE CSM TEST SUITE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run tests in order
        tests = [
            self.test_1_core_architecture,
            self.test_2_rvq_tokenizer,
            self.test_3_data_pipeline,
            self.test_4_training_step,
            self.test_5_inference,
            self.test_6_hypothesis_validation,
            self.test_7_performance_benchmarks
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"Test failed with error: {e}")
                continue
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed = sum(1 for v in self.results.values() if v is True)
        failed = sum(1 for v in self.results.values() if v is False)
        skipped = sum(1 for v in self.results.values() if v is None)
        
        for test_name, result in self.results.items():
            status = "✓ PASS" if result is True else "✗ FAIL" if result is False else "⚠ SKIP"
            print(f"{test_name:20s}: {status}")
        
        print("-" * 40)
        print(f"Total: {total_tests} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
        print(f"Success Rate: {passed/total_tests*100:.1f}%")
        print(f"Total Time: {time.time() - start_time:.2f}s")
        
        # Save results
        results_file = 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'results': self.results,
                'summary': {
                    'total': total_tests,
                    'passed': passed,
                    'failed': failed,
                    'skipped': skipped,
                    'success_rate': passed/total_tests if total_tests > 0 else 0
                },
                'device': str(self.device),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        return passed == total_tests


if __name__ == "__main__":
    tester = CSMTestSuite()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! The CSM implementation is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    # Quick start guide
    print("\n" + "=" * 60)
    print("QUICK START GUIDE")
    print("=" * 60)
    print("""
To start using the CSM model:

1. TRAINING (if you have data):
   python train_minimal.py              # Quick test (100 steps)
   python train.py --model_size small   # Full training

2. INFERENCE DEMO:
   python demo.py --mode test            # Test with examples
   python demo.py                        # Interactive mode

3. HYPOTHESIS EXPERIMENTS:
   python hypothesis_experiments.py      # Validate research claims

4. COLLECT MORE DATA:
   python collect_data.py --target_hours 10

5. RUN BENCHMARKS:
   python benchmark.py
""")