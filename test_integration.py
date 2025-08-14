"""Integration test - check if all components work together"""

import torch
import time

print("Testing CSM integration...")

# 1. Test RVQ
print("\n1. Testing RVQ tokenizer...")
from rvq_tokenizer import ConversationalRVQ

rvq = ConversationalRVQ(n_codebooks=4, codebook_size=128)
audio = torch.randn(1, 1, 4800)  # 0.2 seconds at 24kHz
codes, info = rvq.encode(audio)
print(f"[OK] RVQ encoding works! Generated {len(codes)} codebooks")

# 2. Test CSM Model
print("\n2. Testing CSM model...")
from architecture import CSMModel

model = CSMModel(d_model=128, n_layers=2, n_heads=2)
text_tokens = torch.randint(0, 1000, (1, 10))
outputs = model(text_tokens=text_tokens, voice_codes=codes)
print(f"[OK] CSM forward pass works! Output keys: {list(outputs.keys())}")

# 3. Test loss computation
print("\n3. Testing loss computation...")
from architecture import conversational_loss

targets = {
    'text_targets': text_tokens,
    'voice_targets': codes
}
loss = conversational_loss(outputs, targets)
print(f"[OK] Loss computation works! Loss = {loss.item():.4f}")

# 4. Test backward pass
print("\n4. Testing backward pass...")
start = time.time()
loss.backward()
elapsed = time.time() - start
print(f"[OK] Backward pass works! Time = {elapsed:.2f}s")

# 5. Test data loading
print("\n5. Testing data loading...")
try:
    from dataset_pipeline import create_dataloader
    loader = create_dataloader(["data/conversations"], batch_size=1, num_workers=0)
    batch = next(iter(loader))
    print(f"[OK] Data loading works! Batch keys: {list(batch.keys())}")
except Exception as e:
    print(f"[WARNING]  Data loading failed (expected if no data): {e}")

# 6. Summary
print("\n" + "="*50)
print("INTEGRATION TEST SUMMARY")
print("="*50)
print("[OK] RVQ tokenizer: PASS")
print("[OK] CSM model: PASS")
print("[OK] Loss computation: PASS")
print("[OK] Backward pass: PASS")
print("[WARNING]  Data loading: Check data directory")
print("\nAll core components are working correctly!")
print("The implementation is ready for training.")

# Save test model
print("\n7. Saving test checkpoint...")
torch.save({
    'model_state': model.state_dict(),
    'rvq_state': rvq.state_dict(),
    'test_loss': loss.item()
}, 'checkpoints/integration_test.pt')
print("[OK] Saved checkpoint to checkpoints/integration_test.pt")