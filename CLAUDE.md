# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a research replication project focused on Sesame's "Crossing the Uncanny Valley of Conversational Voice" research. The repository contains a complete implementation of the Conversational Speech Model (CSM) with training infrastructure, data pipeline, and evaluation framework.

## Quick Start

```bash
# Windows
run_experiment.bat

# Linux/Mac
./run_experiment.sh

# Or manually:
python setup_environment.py
python collect_data.py --target_hours 1
python train_minimal.py
python demo.py --mode test
```

## Project Structure

- `architecture.py` - Core CSM model with voice-text cross-attention
- `rvq_tokenizer.py` - Hierarchical RVQ for voice tokenization
- `dataset_pipeline.py` - Conversational audio data handling
- `evaluation_framework.py` - Beyond-traditional metrics for conversation
- `train.py` - Production training script with distributed support
- `demo.py` - Interactive demo for testing the model
- `collect_data.py` - Download and prepare conversational audio

## Key Commands

### Training
```bash
# Minimal test (100 steps)
python train_minimal.py

# Full training
python train.py --model_size tiny --max_steps 10000

# Multi-GPU training
torchrun --nproc_per_node=4 train.py --model_size small
```

### Data Collection
```bash
# Collect 10 hours for testing
python collect_data.py --target_hours 10

# Collect from specific sources
python collect_data.py --sources librispeech common_voice --target_hours 100
```

### Inference
```bash
# Interactive demo
python demo.py

# Test examples
python demo.py --mode test

# Batch processing
python demo.py --mode batch --input conversations.json
```

## Key Technical Concepts

When working on this project, focus on:
- Multimodal transformer architectures that process text and audio tokens together
- RVQ-based audio tokenization with semantic-acoustic hierarchy
- Two-stage generation processes (semantic understanding + acoustic detail)
- Conversational memory system with different decay rates for content vs style
- Evaluation metrics beyond WER: character consistency, emotional coherence, prosodic naturalness

## Model Architecture

The CSM model features:
- **VoiceTextCrossAttention**: Asymmetric attention for different temporal scales
- **ConversationalMemory**: Dual memory system (semantic + prosodic)
- **Hierarchical RVQ**: First 10 codebooks for semantics, rest for acoustics
- **Unified Generation**: Text and voice generated jointly, not separately

## Current Status

- ✅ Complete architecture implementation
- ✅ RVQ tokenizer with conversational features
- ✅ Data pipeline for multiple sources
- ✅ Training infrastructure with distributed support
- ✅ Evaluation framework with novel metrics
- ✅ Interactive demo
- 🔄 Need to collect more data (target: 100+ hours)
- 🔄 Need to run full training experiments
- 🔄 Need to validate key hypotheses