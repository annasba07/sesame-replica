"""
Advanced RVQ Tokenizer Implementation
Based on insights from SoundStream, Encodec, and Mimi

Key insight: Voice is not just compressed audio - it's structured information
with semantic, prosodic, and identity components that should be encoded hierarchically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import math


class CausalConv1d(nn.Module):
    """Causal 1D convolution with proper padding"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) * stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual block with causal convolutions"""
    
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size)
        self.conv2 = CausalConv1d(channels, channels, kernel_size)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + residual


class AudioEncoder(nn.Module):
    """
    Encode raw audio to latent representation
    Innovation: Separate pathways for semantic and acoustic features
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 512,
        channels: List[int] = [32, 64, 128, 256, 512],
        strides: List[int] = [2, 2, 2, 2, 2],
        n_residual_blocks: int = 2
    ):
        super().__init__()
        
        # Initial projection
        self.input_conv = CausalConv1d(input_channels, channels[0], 7)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        in_channels = channels[0]
        
        for out_channels, stride in zip(channels, strides):
            block = nn.Sequential(
                CausalConv1d(in_channels, out_channels, 2 * stride, stride),
                nn.GroupNorm(1, out_channels),
                nn.SiLU(),
                *[ResidualBlock(out_channels) for _ in range(n_residual_blocks)]
            )
            self.encoder_blocks.append(block)
            in_channels = out_channels
        
        # Separate heads for semantic and acoustic features
        self.semantic_head = nn.Sequential(
            nn.Conv1d(channels[-1], latent_dim // 2, 1),
            nn.GroupNorm(1, latent_dim // 2),
            nn.Tanh()  # Bounded semantic features
        )
        
        self.acoustic_head = nn.Sequential(
            nn.Conv1d(channels[-1], latent_dim // 2, 1),
            nn.GroupNorm(1, latent_dim // 2)
        )
        
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_conv(audio)
        
        # Encode through blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        # Split into semantic and acoustic
        semantic = self.semantic_head(x)
        acoustic = self.acoustic_head(x)
        
        # Concatenate but keep track of which is which
        latent = torch.cat([semantic, acoustic], dim=1)
        
        return latent, (semantic, acoustic)


class HierarchicalVectorQuantizer(nn.Module):
    """
    Hierarchical VQ with semantic priors
    Key innovation: Early codebooks capture meaning, later ones capture detail
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        n_codebooks: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 32,
        semantic_codebooks: int = 10,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.semantic_codebooks = semantic_codebooks
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        
        # Project to codebook dimension
        self.input_proj = nn.Conv1d(latent_dim, codebook_dim, 1)
        
        # Codebooks with different initialization strategies
        self.codebooks = nn.ParameterList()
        self.codebook_usage = []  # Track usage for codebook collapse prevention
        
        for i in range(n_codebooks):
            if i < semantic_codebooks:
                # Semantic codebooks: initialized with structure
                codebook = nn.Parameter(torch.randn(codebook_size, codebook_dim) * 0.1)
                # Add some structure to semantic codebooks
                codebook.data[::2] *= -1  # Create opposites
            else:
                # Acoustic codebooks: random initialization
                codebook = nn.Parameter(torch.randn(codebook_size, codebook_dim) * 0.02)
            
            self.codebooks.append(codebook)
            self.codebook_usage.append(torch.zeros(codebook_size))
        
        # EMA updates for codebooks
        self.register_buffer('ema_cluster_size', torch.zeros(n_codebooks, codebook_size))
        self.register_buffer('ema_w', torch.zeros(n_codebooks, codebook_size, codebook_dim))
        
        # Learnable temperature for Gumbel-softmax
        self.temperature = nn.Parameter(torch.ones(n_codebooks) * 1.0)
        
    def forward(
        self,
        latent: torch.Tensor,
        semantic_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], dict]:
        batch_size, _, time_steps = latent.shape
        
        # Project input
        x = self.input_proj(latent)  # [B, codebook_dim, T]
        x = x.transpose(1, 2)  # [B, T, codebook_dim]
        
        # Fix shape issue if there's an extra channel dimension
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        
        quantized = torch.zeros_like(x)
        indices = []
        losses = {
            'commitment': 0.0,
            'codebook': 0.0,
            'entropy': 0.0
        }
        
        residual = x
        
        for i, codebook in enumerate(self.codebooks):
            # For semantic codebooks, apply mask if provided
            if i < self.semantic_codebooks and semantic_mask is not None:
                residual = residual * semantic_mask.unsqueeze(-1)
            
            # Compute distances
            # residual: [B, T, codebook_dim], codebook: [codebook_size, codebook_dim]
            # cdist expects both inputs to have same number of dimensions
            # If residual is [B, T, D] and codebook is [K, D], we want distances to be [B, T, K]
            # Fix shape issue if residual has extra dimensions
            if residual.dim() > 3:
                residual = residual.squeeze(1)
            
            distances = torch.cdist(residual, codebook)  # [B, T, codebook_size]
            
            # Gumbel-softmax for differentiable quantization during training
            if self.training:
                # Add Gumbel noise for exploration
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(distances) + 1e-8) + 1e-8)
                distances_gumbel = -distances / self.temperature[i] + gumbel_noise
                
                # Soft quantization
                soft_indices = F.softmax(distances_gumbel, dim=-1)  # [B, T, codebook_size]
                quantized_step = torch.matmul(soft_indices, codebook)  # [B, T, codebook_dim]
                
                # Hard quantization for indices
                hard_indices = distances.argmin(dim=-1)
                indices.append(hard_indices)
                
                # Entropy regularization to encourage codebook usage
                entropy = -(soft_indices * torch.log(soft_indices + 1e-8)).sum(dim=-1).mean()
                losses['entropy'] += entropy * 0.01
                
                # Update EMA
                if i < self.semantic_codebooks:  # Only update semantic codebooks with EMA
                    # encodings shape: [batch_size, time_steps, codebook_size]
                    encodings = F.one_hot(hard_indices, self.codebook_size).float()
                    
                    # Sum over all batch and time dimensions
                    # Handle case where encodings might have extra dimensions
                    sum_dims = list(range(encodings.dim() - 1))  # All dims except the last (codebook_size)
                    encodings_sum = encodings.sum(dim=sum_dims)  # shape: [codebook_size]
                    
                    self.ema_cluster_size[i] = (
                        self.ema_decay * self.ema_cluster_size[i] +
                        (1 - self.ema_decay) * encodings_sum
                    )
                    
                    # Reshape for matrix multiplication
                    # encodings: [batch_size, time_steps, codebook_size] -> [batch_size * time_steps, codebook_size]
                    encodings_flat = encodings.reshape(-1, self.codebook_size)
                    # residual: [batch_size, time_steps, codebook_dim] -> [batch_size * time_steps, codebook_dim]
                    residual_flat = residual.reshape(-1, self.codebook_dim)
                    
                    # Compute weighted sum of residuals
                    dw = torch.matmul(encodings_flat.t(), residual_flat)  # [codebook_size, codebook_dim]
                    
                    self.ema_w[i, :, :] = (
                        self.ema_decay * self.ema_w[i, :, :] +
                        (1 - self.ema_decay) * dw
                    )
                    
                    # Update codebook
                    n = self.ema_cluster_size[i].sum()
                    cluster_size = (
                        (self.ema_cluster_size[i] + 1e-5) /
                        (n + self.codebook_size * 1e-5) * n
                    )
                    codebook.data = self.ema_w[i] / cluster_size.unsqueeze(1)
                
            else:
                # Hard quantization for inference
                hard_indices = distances.argmin(dim=-1)
                indices.append(hard_indices)
                quantized_step = codebook[hard_indices]
            
            # Update quantized and residual
            quantized = quantized + quantized_step
            residual = residual - quantized_step.detach()
            
            # Commitment loss
            if self.training:
                # Ensure x has the same shape as quantized_step for loss calculation
                if x.shape != quantized_step.shape:
                    # If x has extra dimensions from the original issue, use the squeezed residual shape
                    x_for_loss = residual + quantized_step.detach()
                else:
                    x_for_loss = x
                losses['commitment'] += F.mse_loss(quantized_step.detach(), x_for_loss)
                losses['codebook'] += F.mse_loss(quantized_step, x_for_loss.detach())
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        # Transpose back
        quantized = quantized.transpose(1, 2)  # [B, codebook_dim, T]
        
        # Scale losses
        losses['commitment'] *= self.commitment_weight
        losses['codebook'] *= self.commitment_weight
        
        return quantized, indices, losses


class AudioDecoder(nn.Module):
    """
    Decode quantized representation back to audio
    Innovation: Separate decoding paths that merge at multiple scales
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        output_channels: int = 1,
        channels: List[int] = [512, 256, 128, 64, 32],
        strides: List[int] = [2, 2, 2, 2, 2],
        n_residual_blocks: int = 2
    ):
        super().__init__()
        
        # Initial projection
        self.input_conv = nn.Conv1d(latent_dim, channels[0], 1)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        in_channels = channels[0]
        
        for i, (out_channels, stride) in enumerate(zip(channels[1:], strides)):
            block = nn.Sequential(
                *[ResidualBlock(in_channels) for _ in range(n_residual_blocks)],
                nn.ConvTranspose1d(
                    in_channels, out_channels,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=stride // 2
                ),
                nn.GroupNorm(1, out_channels),
                nn.SiLU()
            )
            self.decoder_blocks.append(block)
            in_channels = out_channels
        
        # Final projection
        self.output_conv = nn.Conv1d(channels[-1], output_channels, 7, padding=3)
        
    def forward(self, quantized: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(quantized)
        
        for block in self.decoder_blocks:
            x = block(x)
        
        audio = self.output_conv(x)
        return torch.tanh(audio)  # Bounded output


class MimiStyleRVQ(nn.Module):
    """
    Complete RVQ system inspired by Mimi
    Key principle: Treat voice as information, not just signal
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        latent_dim: int = 512,
        n_codebooks: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 32,
        semantic_codebooks: int = 10,
        channels: List[int] = [32, 64, 128, 256, 512]
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_codebooks = n_codebooks
        self.semantic_codebooks = semantic_codebooks
        
        # Components
        self.encoder = AudioEncoder(
            input_channels=1,
            latent_dim=latent_dim,
            channels=channels
        )
        
        self.quantizer = HierarchicalVectorQuantizer(
            latent_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            semantic_codebooks=semantic_codebooks
        )
        
        self.decoder = AudioDecoder(
            latent_dim=codebook_dim,
            output_channels=1,
            channels=channels[::-1]  # Reverse for decoder
        )
        
        # Perceptual loss components
        self.mel_spec = nn.Sequential(
            nn.Conv1d(1, 64, 1024, 256, padding=512),
            nn.SiLU(),
            nn.Conv1d(64, 128, 41, 1, padding=20),
            nn.SiLU()
        )
        
    def encode(self, audio: torch.Tensor) -> Tuple[List[torch.Tensor], dict]:
        """Encode audio to discrete codes"""
        # Encode to latent
        latent, (semantic, acoustic) = self.encoder(audio)
        
        # Create semantic mask based on energy in semantic features
        semantic_energy = semantic.abs().mean(dim=1, keepdim=True)
        semantic_mask = (semantic_energy > semantic_energy.mean()).float()
        
        # Quantize
        quantized, indices, losses = self.quantizer(latent, semantic_mask)
        
        return indices, {
            'quantized': quantized,
            'semantic': semantic,
            'acoustic': acoustic,
            'losses': losses
        }
    
    def decode(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """Decode discrete codes to audio"""
        batch_size = indices[0].size(0)
        time_steps = indices[0].size(1)
        
        # Reconstruct quantized representation
        quantized = torch.zeros(
            batch_size, self.quantizer.codebook_dim, time_steps,
            device=indices[0].device
        )
        
        for i, (idx, codebook) in enumerate(zip(indices, self.quantizer.codebooks)):
            quantized += codebook[idx].transpose(1, 2)
        
        # Decode to audio
        audio = self.decoder(quantized)
        
        return audio
    
    def forward(
        self,
        audio: torch.Tensor,
        return_codes: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """Full forward pass"""
        # Encode
        indices, encode_info = self.encode(audio)
        
        # Decode
        reconstructed = self.decode(indices)
        
        # Compute losses
        losses = encode_info['losses'].copy()
        
        # Reconstruction loss
        losses['reconstruction'] = F.mse_loss(reconstructed, audio)
        
        # Perceptual loss
        mel_real = self.mel_spec(audio)
        mel_reconstructed = self.mel_spec(reconstructed)
        losses['perceptual'] = F.l1_loss(mel_reconstructed, mel_real)
        
        # Total loss
        losses['total'] = (
            losses['reconstruction'] +
            losses['perceptual'] * 0.1 +
            losses['commitment'] +
            losses['codebook'] +
            losses['entropy']
        )
        
        info = {
            'losses': losses,
            'semantic': encode_info['semantic'],
            'acoustic': encode_info['acoustic']
        }
        
        if return_codes:
            info['codes'] = indices
        
        return reconstructed, info
    
    def get_codebook_usage(self) -> List[float]:
        """Monitor codebook utilization"""
        usage = []
        for i in range(self.n_codebooks):
            if i < self.semantic_codebooks:
                # For EMA codebooks
                active = (self.quantizer.ema_cluster_size[i] > 1.0).float().mean()
            else:
                # For standard codebooks (would need to track during forward)
                active = 1.0  # Placeholder
            usage.append(active.item())
        return usage


# Utility functions for training
def compute_discriminator_loss(
    discriminator: nn.Module,
    real_audio: torch.Tensor,
    fake_audio: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute discriminator and generator losses"""
    # Discriminator loss
    real_scores = discriminator(real_audio)
    fake_scores = discriminator(fake_audio.detach())
    
    d_loss = F.relu(1 - real_scores).mean() + F.relu(1 + fake_scores).mean()
    
    # Generator loss
    fake_scores_g = discriminator(fake_audio)
    g_loss = -fake_scores_g.mean()
    
    return d_loss, g_loss


class ConversationalRVQ(MimiStyleRVQ):
    """
    Extended RVQ specifically for conversational audio
    Adds speaker and emotion awareness
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional components for conversation
        # Use the latent_dim from parent class instead of channels
        self.speaker_encoder = nn.Sequential(
            nn.Conv1d(512, 256, 1),  # Assuming default latent_dim=512
            nn.GroupNorm(1, 256),
            nn.SiLU(),
            nn.Conv1d(256, 128, 1),
            nn.Tanh()
        )
        
        self.emotion_encoder = nn.Sequential(
            nn.Conv1d(512, 128, 1),  # Assuming default latent_dim=512
            nn.GroupNorm(1, 128),
            nn.SiLU(),
            nn.Conv1d(128, 64, 1),
            nn.Tanh()
        )
        
    def encode_conversational(
        self,
        audio: torch.Tensor,
        extract_features: bool = True
    ) -> Tuple[List[torch.Tensor], dict]:
        """Encode with conversational features"""
        # Standard encoding
        indices, info = self.encode(audio)
        
        if extract_features:
            # Extract conversational features
            # Get intermediate features from encoder
            x = self.encoder.input_conv(audio)
            features = x
            for block in self.encoder.encoder_blocks:
                features = block(features)
            
            # Extract speaker and emotion
            speaker_embedding = self.speaker_encoder(features).mean(dim=2)
            emotion_embedding = self.emotion_encoder(features).mean(dim=2)
            
            info['speaker'] = speaker_embedding
            info['emotion'] = emotion_embedding
        
        return indices, info


if __name__ == "__main__":
    # Test the implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    rvq = ConversationalRVQ(
        sample_rate=24000,
        n_codebooks=32,
        codebook_size=1024,
        semantic_codebooks=10
    ).to(device)
    
    # Test data
    batch_size = 2
    audio_length = 24000 * 2  # 2 seconds
    audio = torch.randn(batch_size, 1, audio_length).to(device)
    
    # Forward pass
    reconstructed, info = rvq(audio, return_codes=True)
    
    print(f"Input shape: {audio.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Number of codes: {len(info['codes'])}")
    print(f"Code shape: {info['codes'][0].shape}")
    print(f"Losses: {info['losses']}")
    
    # Test conversational features
    indices, conv_info = rvq.encode_conversational(audio)
    print(f"Speaker embedding shape: {conv_info['speaker'].shape}")
    print(f"Emotion embedding shape: {conv_info['emotion'].shape}")
    
    # Check codebook usage
    usage = rvq.get_codebook_usage()
    print(f"Codebook usage: {usage[:5]}...")  # First 5 codebooks