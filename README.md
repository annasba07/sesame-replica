# CSM (Conversational Speech Model) Replication

A research implementation of Sesame's approach to crossing the uncanny valley of conversational voice, creating AI that truly converses rather than just speaks.

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/sesame-replica.git
cd sesame-replica

# Run the complete experiment (setup + data + training + demo)
./run_experiment.sh  # Linux/Mac
run_experiment.bat   # Windows
```

This will:
1. Install all dependencies
2. Download 1 hour of test data
3. Train a tiny model for 100 steps
4. Run demo examples

## 🏗️ Architecture

Our implementation of CSM features three key innovations:

### 1. Voice-Text Cross-Attention
- Asymmetric attention mechanisms for text (local) vs voice (global)
- Allows the model to understand that voice operates on different temporal scales

### 2. Hierarchical RVQ 
- First 10 codebooks capture semantic information
- Remaining 22 codebooks capture acoustic details
- Enables voice tokens to carry meaning, not just sound

### 3. Conversational Memory
- Dual memory system: semantic (what was said) and prosodic (how it was said)
- Different decay rates for content vs. style
- Maintains character consistency across conversations

## 📊 Results

Our evaluation goes beyond traditional metrics (WER, MCD) to measure what matters for conversation:

- **Character Consistency**: Does the voice maintain identity across turns?
- **Emotional Coherence**: Are emotional transitions natural?
- **Prosodic Naturalness**: Does the voice match the semantic content?
- **Conversational Flow**: Is the turn-taking natural?

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the setup script
python setup_environment.py
```

## 📚 Training

### Data Collection
```bash
# Collect conversational audio (adjust hours as needed)
python collect_data.py --target_hours 100
```

### Model Training
```bash
# Train tiny model (350M params)
python train.py --model_size tiny --max_steps 50000

# Train small model (1.5B params) with multiple GPUs
torchrun --nproc_per_node=4 train.py --model_size small

# Resume from checkpoint
python train.py --resume checkpoints/best_model.pt
```

### Monitoring
- Training metrics are logged to Weights & Biases
- View at: https://wandb.ai/your-project/csm-replication

## 🎤 Demo

```bash
# Interactive conversation
python demo.py

# Test with examples
python demo.py --mode test

# Batch processing
python demo.py --mode batch --input conversations.json
```

## 📁 Project Structure

```
sesame-replica/
├── architecture.py          # Core CSM model
├── rvq_tokenizer.py        # Hierarchical voice tokenization  
├── dataset_pipeline.py     # Conversational data handling
├── evaluation_framework.py # Novel evaluation metrics
├── train.py               # Production training script
├── demo.py                # Interactive demo
├── collect_data.py        # Data collection utilities
└── research-papers/       # Reference papers
```

## 🔬 Key Hypotheses

We're testing three core hypotheses:

1. **Voice tokens contain semantic information** - Can we predict words from voice codes?
2. **Prosodic memory improves coherence** - Does remembering *how* things were said help?
3. **Cross-modal attention enables voice-text fusion** - Is unified processing better than separate pipelines?

## 📈 Scaling Roadmap

- [x] Tiny (350M) - Proof of concept
- [ ] Small (1.5B) - Validate hypotheses  
- [ ] Medium (7B) - Approach human quality
- [ ] Large (70B) - Production ready

## 🤝 Contributing

This is a research project aimed at advancing conversational AI. Contributions welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📖 Citations

If you use this code in your research, please cite:

```bibtex
@misc{csm-replication,
  title={Replicating Conversational Speech Models},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/sesame-replica}
}
```

## 🙏 Acknowledgments

- Sesame team for the groundbreaking CSM research
- Authors of RQ-Transformer, AudioLM, and other referenced papers
- Open source community for tools and datasets

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

**Note**: This is a research replication. For production use, please refer to Sesame's official implementation once released.