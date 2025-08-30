"""
CSM Architecture Implementation
Following Ilya's principle: "The architecture should be as simple as possible, but no simpler"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class VoiceTextCrossAttention(nn.Module):
    """
    Key insight: Voice and text operate on different temporal scales.
    Text is discrete and local, voice is continuous and global.
    """
    
    def __init__(self, d_model: int = 4096, n_heads: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Asymmetric projections for text and voice
        self.text_proj = nn.Linear(d_model, d_model * 3)  # Q, K, V
        self.voice_proj = nn.Linear(d_model, d_model * 3)  # Q, K, V
        
        # Learnable temporal bias for voice attention
        self.voice_temporal_bias = nn.Parameter(torch.zeros(1, n_heads, 2048, 2048))
        
        # Cross-modal fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(
        self, 
        text_hidden: torch.Tensor,
        voice_hidden: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        voice_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = text_hidden.size(0)
        
        # Text attention (local, precise)
        text_q, text_k, text_v = self.text_proj(text_hidden).chunk(3, dim=-1)
        text_q = text_q.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        text_k = text_k.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        text_v = text_v.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        text_attn = F.scaled_dot_product_attention(
            text_q, text_k, text_v,
            attn_mask=text_mask,
            dropout_p=0.1 if self.training else 0.0
        )
        text_attn = text_attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Voice attention (global, continuous)
        voice_q, voice_k, voice_v = self.voice_proj(voice_hidden).chunk(3, dim=-1)
        voice_q = voice_q.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        voice_k = voice_k.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        voice_v = voice_v.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        # Add temporal bias to voice attention
        voice_scores = torch.matmul(voice_q, voice_k.transpose(-2, -1)) / math.sqrt(self.d_head)
        seq_len = voice_scores.size(-1)
        voice_scores = voice_scores + self.voice_temporal_bias[:, :, :seq_len, :seq_len]
        
        if voice_mask is not None:
            voice_scores = voice_scores.masked_fill(voice_mask == 0, -1e9)
            
        voice_attn_weights = F.softmax(voice_scores, dim=-1)
        voice_attn = torch.matmul(voice_attn_weights, voice_v)
        voice_attn = voice_attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Cross-modal fusion with gating
        # Handle different sequence lengths - use the text length as reference
        text_len = text_attn.size(1)
        voice_len = voice_attn.size(1)
        
        if text_len != voice_len:
            # Interpolate voice to match text length
            voice_attn = F.interpolate(
                voice_attn.transpose(1, 2), 
                size=text_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        fusion_input = torch.cat([text_attn, voice_attn], dim=-1)
        gate = self.fusion_gate(fusion_input)
        
        # Gated fusion: let the model learn when to use text vs voice
        fused = gate * text_attn + (1 - gate) * voice_attn
        
        return self.out_proj(fused), gate


class ConversationalMemory(nn.Module):
    """
    Dual memory system: semantic (what) and prosodic (how)
    Inspired by Hopfield networks but adapted for continuous voice
    """
    
    def __init__(self, d_model: int = 4096, memory_size: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        
        # Semantic memory (content)
        self.semantic_memory = nn.Parameter(torch.randn(memory_size, d_model))
        self.semantic_decay = nn.Parameter(torch.ones(memory_size) * 0.95)
        
        # Prosodic memory (style/emotion)
        self.prosodic_memory = nn.Parameter(torch.randn(memory_size, d_model))
        self.prosodic_decay = nn.Parameter(torch.ones(memory_size) * 0.8)  # Faster decay
        
        # Memory update networks
        self.semantic_update = nn.GRUCell(d_model, d_model)
        self.prosodic_update = nn.GRUCell(d_model, d_model)
        
        # Attention for memory retrieval
        self.memory_attention = nn.MultiheadAttention(d_model, num_heads=8)
        
    def forward(
        self,
        text_features: torch.Tensor,
        voice_features: torch.Tensor,
        update: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = text_features.size(0)
        
        # Retrieve relevant memories
        semantic_query = text_features.mean(dim=1, keepdim=True)  # [B, 1, D]
        prosodic_query = voice_features.mean(dim=1, keepdim=True)  # [B, 1, D]
        
        # Expand memories for batch processing
        semantic_mem_exp = self.semantic_memory.unsqueeze(0).expand(batch_size, -1, -1)
        prosodic_mem_exp = self.prosodic_memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Retrieve semantic memory
        semantic_retrieved, _ = self.memory_attention(
            semantic_query,
            semantic_mem_exp.transpose(0, 1),
            semantic_mem_exp.transpose(0, 1)
        )
        semantic_retrieved = semantic_retrieved.transpose(0, 1).squeeze(1)
        
        # Retrieve prosodic memory
        prosodic_retrieved, _ = self.memory_attention(
            prosodic_query,
            prosodic_mem_exp.transpose(0, 1),
            prosodic_mem_exp.transpose(0, 1)
        )
        prosodic_retrieved = prosodic_retrieved.transpose(0, 1).squeeze(1)
        
        if update and self.training:
            # Update memories with decay
            self.semantic_memory.data = (
                self.semantic_decay.unsqueeze(1) * self.semantic_memory.data +
                (1 - self.semantic_decay.unsqueeze(1)) * semantic_query.squeeze(1).mean(0)
            )
            self.prosodic_memory.data = (
                self.prosodic_decay.unsqueeze(1) * self.prosodic_memory.data +
                (1 - self.prosodic_decay.unsqueeze(1)) * prosodic_query.squeeze(1).mean(0)
            )
        
        return semantic_retrieved, prosodic_retrieved


class RVQEncoder(nn.Module):
    """
    Residual Vector Quantization for voice encoding
    Key innovation: Hierarchical codebooks with semantic priors
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        codebook_size: int = 1024,
        num_codebooks: int = 32,
        codebook_dim: int = 128
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, codebook_dim)
        
        # Hierarchical codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, codebook_dim)
            for _ in range(num_codebooks)
        ])
        
        # Semantic conditioning for codebooks
        self.semantic_proj = nn.ModuleList([
            nn.Linear(4096, codebook_dim) for _ in range(10)  # First 10 are semantic
        ])
        
        # Commitment loss weight
        self.commitment_weight = 0.25
        
    def forward(
        self,
        audio_features: torch.Tensor,
        semantic_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        batch_size, seq_len, _ = audio_features.shape
        
        # Project input
        x = self.input_proj(audio_features)
        residual = x
        
        quantized = torch.zeros_like(x)
        indices = []
        commitment_loss = 0.0
        
        for i, codebook in enumerate(self.codebooks):
            # Apply semantic conditioning to first 10 codebooks
            if i < 10 and semantic_context is not None:
                semantic_bias = self.semantic_proj[i](semantic_context)
                semantic_bias = semantic_bias.unsqueeze(1).expand(-1, seq_len, -1)
                residual = residual + 0.1 * semantic_bias  # Soft semantic guidance
            
            # Find nearest codebook entries
            distances = torch.cdist(residual, codebook.weight.unsqueeze(0))
            min_indices = distances.argmin(dim=-1)
            indices.append(min_indices)
            
            # Quantize
            quantized_step = codebook(min_indices)
            quantized = quantized + quantized_step
            
            # Update residual
            residual = residual - quantized_step.detach()
            
            # Commitment loss
            commitment_loss += F.mse_loss(quantized_step.detach(), x)
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        return quantized, indices, commitment_loss * self.commitment_weight


class CSMModel(nn.Module):
    """
    Conversational Speech Model
    Philosophy: Voice is not just sound, it's a carrier of meaning
    """
    
    def __init__(
        self,
        vocab_size: int = 128256,  # Llama 3 vocab
        d_model: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        max_seq_len: int = 2048,
        rvq_codebooks: int = 32,
        rvq_codebook_size: int = 1024
    ):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Token embeddings
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.voice_embed = nn.ModuleList([
            nn.Embedding(rvq_codebook_size, d_model // rvq_codebooks)
            for _ in range(rvq_codebooks)
        ])
        
        # Positional encoding with learned modality-specific biases
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.text_modal_bias = nn.Parameter(torch.zeros(1, 1, d_model))
        self.voice_modal_bias = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Conversational memory
        self.memory = ConversationalMemory(d_model)
        
        # Transformer layers with cross-modal attention
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'cross_attn': VoiceTextCrossAttention(d_model, n_heads),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.SiLU(),
                    nn.Linear(d_model * 4, d_model)
                ),
                'ln1': nn.LayerNorm(d_model),
                'ln2': nn.LayerNorm(d_model),
                'ln3': nn.LayerNorm(d_model)
            })
            for _ in range(n_layers)
        ])
        
        # Output heads
        self.text_head = nn.Linear(d_model, vocab_size)
        self.voice_head = nn.ModuleList([
            nn.Linear(d_model, rvq_codebook_size)
            for _ in range(rvq_codebooks)
        ])
        
        # RVQ encoder for training
        self.rvq_encoder = RVQEncoder(
            input_dim=512,  # Assume mel-spectrogram features
            codebook_size=rvq_codebook_size,
            num_codebooks=rvq_codebooks
        )
        
        self._init_weights()
        
    def _init_weights(self):
        # Xavier initialization with careful scaling
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(
        self,
        text_tokens: Optional[torch.Tensor] = None,
        voice_codes: Optional[List[torch.Tensor]] = None,
        raw_audio: Optional[torch.Tensor] = None,
        conversation_history: Optional[List[dict]] = None
    ) -> dict:
        batch_size = text_tokens.size(0) if text_tokens is not None else voice_codes[0].size(0)
        device = text_tokens.device if text_tokens is not None else voice_codes[0].device
        
        # Process text tokens
        text_hidden = None
        if text_tokens is not None:
            text_hidden = self.text_embed(text_tokens)
            positions = torch.arange(text_tokens.size(1), device=device)
            text_hidden = text_hidden + self.pos_embed(positions) + self.text_modal_bias
        
        # Process voice codes or raw audio
        voice_hidden = None
        rvq_loss = 0.0
        if raw_audio is not None and self.training:
            # Training: encode raw audio
            voice_hidden, voice_codes, rvq_loss = self.rvq_encoder(
                raw_audio, 
                semantic_context=text_hidden.mean(dim=1) if text_hidden is not None else None
            )
        elif voice_codes is not None:
            # Inference: use provided codes
            voice_embeds = []
            for i, (code, embed_layer) in enumerate(zip(voice_codes, self.voice_embed)):
                voice_embeds.append(embed_layer(code))
            voice_hidden = torch.cat(voice_embeds, dim=-1)
            
            # Project to d_model if needed
            if voice_hidden.size(-1) != self.d_model:
                # Create projection layer if not exists
                if not hasattr(self, 'voice_projection'):
                    self.voice_projection = nn.Linear(voice_hidden.size(-1), self.d_model).to(voice_hidden.device)
                voice_hidden = self.voice_projection(voice_hidden)
            seq_len = voice_codes[0].size(1)
            positions = torch.arange(seq_len, device=device)
            # Clamp positions to max_seq_len to avoid index out of range
            positions = positions.clamp(max=self.max_seq_len - 1)
            voice_hidden = voice_hidden + self.pos_embed(positions) + self.voice_modal_bias
        
        # Retrieve conversational memory
        if text_hidden is not None and voice_hidden is not None:
            semantic_mem, prosodic_mem = self.memory(text_hidden, voice_hidden)
            # Inject memory into first layer
            text_hidden[:, 0] = text_hidden[:, 0] + 0.1 * semantic_mem
            voice_hidden[:, 0] = voice_hidden[:, 0] + 0.1 * prosodic_mem
        
        # Process through transformer layers
        for layer in self.layers:
            # Self-attention for each modality
            if text_hidden is not None:
                text_hidden = text_hidden + layer['self_attn'](
                    layer['ln1'](text_hidden),
                    layer['ln1'](text_hidden),
                    layer['ln1'](text_hidden)
                )[0]
            
            if voice_hidden is not None:
                voice_hidden = voice_hidden + layer['self_attn'](
                    layer['ln1'](voice_hidden),
                    layer['ln1'](voice_hidden),
                    layer['ln1'](voice_hidden)
                )[0]
            
            # Cross-modal attention
            if text_hidden is not None and voice_hidden is not None:
                fused, gate = layer['cross_attn'](
                    layer['ln2'](text_hidden),
                    layer['ln2'](voice_hidden)
                )
                # Residual connection with learnable weighting
                text_hidden = text_hidden + 0.5 * fused
                
                # For voice, interpolate fused back to voice length if needed
                if fused.size(1) != voice_hidden.size(1):
                    fused_for_voice = F.interpolate(
                        fused.transpose(1, 2),
                        size=voice_hidden.size(1),
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)
                else:
                    fused_for_voice = fused
                voice_hidden = voice_hidden + 0.5 * fused_for_voice
            
            # FFN
            if text_hidden is not None:
                text_hidden = text_hidden + layer['ffn'](layer['ln3'](text_hidden))
            if voice_hidden is not None:
                voice_hidden = voice_hidden + layer['ffn'](layer['ln3'](voice_hidden))
        
        # Generate outputs
        outputs = {}
        
        if text_hidden is not None:
            outputs['text_logits'] = self.text_head(text_hidden)
        
        if voice_hidden is not None:
            outputs['voice_logits'] = [head(voice_hidden) for head in self.voice_head]
        
        outputs['rvq_loss'] = rvq_loss
        
        return outputs


def conversational_loss(outputs: dict, targets: dict, context: Optional[dict] = None) -> torch.Tensor:
    """
    Loss function that understands conversation is more than words
    """
    total_loss = 0.0
    
    # Text loss (standard cross-entropy)
    if 'text_logits' in outputs and 'text_targets' in targets:
        text_loss = F.cross_entropy(
            outputs['text_logits'].reshape(-1, outputs['text_logits'].size(-1)),
            targets['text_targets'].reshape(-1)
        )
        total_loss += text_loss
    
    # Voice loss (multi-codebook)
    if 'voice_logits' in outputs and 'voice_targets' in targets:
        voice_loss = 0.0
        for i, (logits, target) in enumerate(zip(outputs['voice_logits'], targets['voice_targets'])):
            # Weight early codebooks more heavily (they carry semantic info)
            weight = 1.0 if i < 10 else 0.5
            voice_loss += weight * F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1)
            )
        total_loss += voice_loss / len(outputs['voice_logits'])
    
    # RVQ commitment loss
    if 'rvq_loss' in outputs:
        total_loss += outputs['rvq_loss']
    
    # Novel: Conversational coherence loss
    if context is not None and 'speaker_embeddings' in context:
        # Ensure generated voice maintains speaker identity
        # This would require additional speaker verification module
        pass
    
    return total_loss


if __name__ == "__main__":
    # Test the architecture
    model = CSMModel()
    
    # Dummy inputs
    batch_size = 2
    seq_len = 512
    text_tokens = torch.randint(0, 128256, (batch_size, seq_len))
    voice_codes = [torch.randint(0, 1024, (batch_size, seq_len)) for _ in range(32)]
    
    # Forward pass
    outputs = model(text_tokens=text_tokens, voice_codes=voice_codes)
    
    print(f"Text logits shape: {outputs['text_logits'].shape}")
    print(f"Voice logits shapes: {[logits.shape for logits in outputs['voice_logits']]}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")