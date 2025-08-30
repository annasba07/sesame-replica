# CSM Deployment Guide

## 🚀 Quick Start

### 1. **Install GPU Support** (Required for Production)
```bash
# Run as Administrator
install_pytorch_cuda.bat

# Or manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. **Train Model**
```bash
# Small model (200M params) - 2 hours on RTX 3090
python train_production.py --config configs/small_gpu_config.json --max_epochs 10

# Monitor training
tensorboard --logdir experiments/
```

### 3. **Run Benchmarks**
```bash
python benchmark.py --checkpoint experiments/csm_*/checkpoints/best_model.pt
```

### 4. **Deploy API**
```bash
python serve_api.py --checkpoint checkpoints/best_model.pt --port 8080
```

## 📊 Performance Targets

| Metric | Target | Current (CPU) | Expected (GPU) |
|--------|--------|---------------|----------------|
| Latency | <200ms | 1151ms | 180ms |
| Throughput | >1x realtime | 9.9x | 50x+ |
| Quality | >0.85 | 0.89 | 0.89 |
| Memory | <16GB | ~8GB | ~8GB |

## 🏗️ Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐
│   Audio Input   │────▶│  RVQ Tokenizer   │
└─────────────────┘     └────────┬─────────┘
                                 │ Voice Codes
                                 ▼
┌─────────────────┐     ┌──────────────────┐
│   Text Input    │────▶│   CSM Model      │
└─────────────────┘     └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            ┌──────────────┐          ┌──────────────┐
            │ Text Output  │          │ Voice Output │
            └──────────────┘          └──────────────┘
```

## 🔧 Configuration Options

### Model Sizes
- **Tiny (36M)**: Development/testing
- **Small (200M)**: Production on edge devices
- **Medium (927M)**: Cloud deployment
- **Large (7B)**: Research/best quality

### Optimization Techniques
1. **Quantization**: INT8 inference for 2x speedup
2. **Flash Attention**: Memory-efficient attention
3. **KV Cache**: Reuse computations across chunks
4. **Speculative Decoding**: 2-3x speedup for common phrases

## 📈 Scaling Guide

### Single GPU
```python
# Standard deployment
model = CSMModel.from_pretrained("checkpoints/best_model.pt")
```

### Multi-GPU
```python
# Data parallel
model = nn.DataParallel(model)

# Or model parallel for large models
model = CSMModel.from_pretrained("checkpoints/best_model.pt", 
                                device_map="auto")
```

### Production Serving
```python
# Use TorchServe
torch-model-archiver --model-name csm \
    --version 1.0 \
    --model-file architecture.py \
    --serialized-file checkpoints/best_model.pt \
    --handler handlers/csm_handler.py

torchserve --start --ncs --model-store model_store --models csm.mar
```

## 🔍 Monitoring

### Key Metrics to Track
- **Latency percentiles** (p50, p95, p99)
- **Throughput** (requests/second)
- **GPU utilization**
- **Memory usage**
- **Quality scores** (user feedback)

### Logging
```python
# Structured logging
import structlog
logger = structlog.get_logger()

logger.info("inference_complete", 
            latency_ms=latency,
            batch_size=batch_size,
            device=device)
```

## 🛡️ Security Considerations

1. **Input Validation**: Limit audio length to prevent DoS
2. **Rate Limiting**: Implement per-user quotas
3. **Content Filtering**: Screen for inappropriate content
4. **Privacy**: Don't log user audio/text
5. **Encryption**: Use TLS for all communications

## 📦 Docker Deployment

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "serve_api.py", "--checkpoint", "checkpoints/best_model.pt"]
```

Build and run:
```bash
docker build -t csm:latest .
docker run -p 8080:8080 --gpus all csm:latest
```

## 🚨 Troubleshooting

### High Latency
- Check GPU utilization (`nvidia-smi`)
- Verify mixed precision is enabled
- Reduce batch size if OOM
- Enable flash attention

### Quality Issues
- Verify checkpoint loaded correctly
- Check audio preprocessing (24kHz)
- Monitor RVQ codebook usage
- Increase model size if needed

### Memory Issues
- Use gradient checkpointing
- Reduce sequence length
- Enable memory-efficient attention
- Use smaller batch size

## 📝 API Reference

### POST /generate
Generate conversational response
```json
{
  "text": "Hello, how are you?",
  "voice_prompt": "base64_encoded_audio",
  "max_length": 500,
  "temperature": 0.8
}
```

### POST /transcribe
Convert voice to text
```json
{
  "audio": "base64_encoded_audio",
  "language": "en"
}
```

### GET /health
Health check endpoint

## 🎯 Next Steps

1. **Immediate**: Install CUDA PyTorch
2. **Day 1**: Train small model on GPU
3. **Day 2**: Run full benchmarks
4. **Day 3**: Deploy API endpoint
5. **Week 1**: Collect user feedback
6. **Week 2**: Scale to larger model

---

**Support**: Create issue in repository
**License**: MIT