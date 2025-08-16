# CSM Replication Research Plan

## Core Hypothesis
Voice contains semantic information that text cannot capture. A truly conversational AI must process voice as a primary modality, not as a post-processing step.

## Phase 1: Theoretical Foundation (Weeks 1-2)

### 1.1 Unified Representation Theory
- **Key insight**: Text and voice are projections of a higher-dimensional semantic space
- **Approach**: Design a shared embedding space where prosody, emotion, and meaning coexist
- **Validation**: Prove that voice embeddings can predict semantic relationships better than text alone

### 1.2 Conversational Memory Architecture
- **Problem**: Current models lack persistent voice identity across turns
- **Solution**: Implement a dual-memory system:
  - Semantic memory (what was said)
  - Prosodic memory (how it was said)
- **Innovation**: Memory vectors that decay differently for content vs. style

## Phase 2: Minimal Viable Architecture (Weeks 3-6)

### 2.1 RVQ Implementation
```python
# Core architecture sketch
class VoiceAwareTransformer:
    def __init__(self):
        self.text_encoder = LlamaTokenizer()
        self.voice_encoder = RVQEncoder(layers=32, codebook_size=1024)
        self.unified_transformer = nn.Transformer(
            d_model=4096,
            nhead=32,
            num_layers=32,
            # Key innovation: asymmetric attention
            custom_attention=VoiceTextCrossAttention()
        )
```

### 2.2 Two-Stage Generation Pipeline
1. **Stage 1**: Semantic planning (what to say + how to say it)
   - Input: conversation history + current text
   - Output: semantic tokens + prosodic sketch
   
2. **Stage 2**: Acoustic realization
   - Input: Stage 1 output
   - Output: Full RVQ codes

### 2.3 Compute Optimization Strategy
- **Problem**: Full RVQ processing is memory-intensive
- **Solution**: Hierarchical processing
  - Critical codes (0-10): Always process
  - Detail codes (11-31): Stochastic selection during training
  - Inference: Generate all codes but cache aggressively

## Phase 3: Dataset Engineering (Weeks 4-8, parallel)

### 3.1 Data Requirements
- **Scale**: 100K hours for proof-of-concept (vs. Sesame's 1M)
- **Quality**: Conversational data with emotional range
- **Sources**:
  - Podcasts (natural conversation)
  - Audiobooks (emotional range)
  - Movie dialogues (character consistency)

### 3.2 Data Pipeline
```python
class ConversationalDataset:
    def process_conversation(self, audio, transcript):
        # Extract conversational segments
        turns = self.segment_dialogue(audio, transcript)
        
        # Compute conversational features
        for i, turn in enumerate(turns):
            turn.context = turns[max(0, i-5):i]  # 5-turn context
            turn.prosodic_features = self.extract_prosody(turn.audio)
            turn.semantic_features = self.extract_semantics(turn.text)
            
        return turns
```

## Phase 4: Novel Evaluation Framework (Weeks 7-9)

### 4.1 Beyond Traditional Metrics
- **Conversational Coherence Score**: Does the voice maintain personality?
- **Emotional Continuity**: Do emotional states evolve naturally?
- **Pragmatic Success**: Does the model achieve conversational goals?

### 4.2 Adversarial Testing
```python
class ConversationalAdversary:
    def test_identity_preservation(self, model):
        # Can the model maintain voice identity under pressure?
        contexts = [
            "friendly_chat",
            "heated_argument", 
            "technical_explanation",
            "emotional_confession"
        ]
        return measure_voice_consistency(model, contexts)
```

## Phase 5: Scaling Strategy (Weeks 10-12)

### 5.1 Model Scaling Experiments
- Start with 350M parameters (tiny)
- Scale to 1.5B (small) 
- Target 7B (medium) for publication

### 5.2 Distributed Training Architecture
- Data parallelism for transformer backbone
- Model parallelism for RVQ decoder
- Gradient checkpointing for memory efficiency

## Key Technical Innovations

### 1. Attention Mechanism Redesign
```python
class VoiceTextCrossAttention(nn.Module):
    """
    Asymmetric attention where voice tokens attend to a wider
    temporal context than text tokens
    """
    def forward(self, text_tokens, voice_tokens):
        # Text attends locally (semantic precision)
        text_attn = self.local_attention(text_tokens, window=256)
        
        # Voice attends globally (prosodic continuity)
        voice_attn = self.global_attention(voice_tokens, window=2048)
        
        # Cross-modal binding
        return self.bind(text_attn, voice_attn)
```

### 2. Loss Function Design
```python
def conversational_loss(pred, target, context):
    # Standard reconstruction loss
    recon_loss = F.mse_loss(pred.audio, target.audio)
    
    # Semantic alignment loss
    semantic_loss = cosine_loss(pred.semantics, target.semantics)
    
    # Novel: Conversational dynamics loss
    dynamics_loss = measure_prosodic_evolution(pred, target, context)
    
    # Novel: Character consistency loss
    character_loss = voice_identity_loss(pred, context.speaker_history)
    
    return recon_loss + semantic_loss + dynamics_loss + character_loss
```

### 3. Inference Optimization
- Speculative decoding for voice tokens
- KV-cache sharing between text and voice streams
- Adaptive quality: trade fidelity for latency based on conversation pace

## Critical Experiments

### Experiment 1: Voice-Semantic Coupling
**Hypothesis**: Voice tokens contain semantic information
**Test**: Train probe to predict next word from voice tokens alone
**Success metric**: >70% accuracy on common words

### Experiment 2: Conversational Memory
**Hypothesis**: Prosodic memory improves coherence
**Test**: Compare models with/without prosodic memory on multi-turn dialogues
**Success metric**: 2x improvement in character consistency scores

### Experiment 3: Emotional Dynamics
**Hypothesis**: Model can generate appropriate emotional transitions
**Test**: Generate responses to emotional prompts, measure naturalness
**Success metric**: Human evaluators prefer our model 60% of the time

## Risk Mitigation

1. **Data Quality**: Partner with speech researchers for high-quality annotations
2. **Compute Constraints**: Start with smaller models, prove concept before scaling
3. **Evaluation Validity**: Develop rigorous human evaluation protocols early

## Success Criteria

- Generate 30-second conversational responses indistinguishable from human speech
- Maintain character consistency across 10+ conversation turns
- Demonstrate emotional intelligence in voice generation
- Achieve real-time inference on consumer hardware (with quality tradeoffs)

## Timeline

- Weeks 1-2: Theory and architecture design
- Weeks 3-6: Build core components
- Weeks 4-8: Data pipeline (parallel)
- Weeks 7-9: Evaluation framework
- Weeks 10-12: Training and scaling
- Weeks 13-16: Experimentation and iteration
- Weeks 17-20: Paper writing and open-source release