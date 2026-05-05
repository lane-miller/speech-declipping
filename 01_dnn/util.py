"""
util.py
-------
Shared utilities for DeclipNet training and evaluation.

Contents:
    - DeclipDataset: PyTorch Dataset for pre-generated train/val block tensors
    - DeclipNet: U-Net with dilated conv bottleneck and self-attention for speech declipping
    - Loss functions: weighted waveform L1, multi-resolution STFT, DWT-based losses
    - si_sdr: scale-invariant signal-to-distortion ratio metric
"""

# Imports
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# =============================================================================
# Dataset
# =============================================================================

# Dataset class
class DeclipDataset(Dataset):
    """
    Dataset for pre-generated declipping train/val blocks.
    
    Loads clean blocks from a single .pt tensor file and corresponding
    manifest JSON containing per-block talker ID and clipping threshold alpha.
    Clipping is applied on-the-fly using the stored alpha values.
    
    Args:
        blocks_path: path to .pt file containing clean blocks (N, 1, BS)
        manifest_path: path to JSON manifest with per-block metadata
    """

    def __init__(self, blocks_path: Path, manifest_path: Path):
        self.blocks = torch.load(blocks_path, weights_only=True)
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        assert len(self.blocks) == len(self.manifest), \
            f"Block/manifest length mismatch: {len(self.blocks)} vs {len(self.manifest)}"

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        clean = self.blocks[idx]  # (1, BS)
        alpha = self.manifest[idx]["alpha"]

        peak = clean.abs().max()
        threshold = alpha * peak
        clipped = clean.clamp(-threshold, threshold)

        return clipped, clean

# =============================================================================
# Model
# =============================================================================

# Encoder block
class EncoderBlock(nn.Module):
    """Single encoder block: Conv1d -> ReLU -> Conv1x1 -> GLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=8, stride=2, padding=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, \
                              stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.conv1x1 = nn.Conv1d(out_channels, out_channels * 2, 1)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.glu(self.conv1x1(x))
        return x

# Decoder block
class DecoderBlock(nn.Module):
    """Single decoder block: Conv1x1 -> GLU -> ConvTranspose1d -> ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=8, \
                 stride=2, padding=3, last=False):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, in_channels * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.convtranspose = nn.ConvTranspose1d(in_channels, out_channels, \
                                                kernel_size, stride=stride, padding=padding)
        self.last = last
        self.relu = nn.ReLU()

    def forward(self, x, skip):
        x = x + skip
        x = self.glu(self.conv1x1(x))
        x = self.convtranspose(x)
        if not self.last:
            x = self.relu(x)
        return x

# Dilated conv block
class DilatedBlock(nn.Module):
    """Dilated conv block with residual connection: Conv1d -> ReLU -> + residual"""
    def __init__(self, channels, kernel_size=8, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, 
                              dilation=dilation, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_padded = F.pad(x, (self.padding, 0))
        return self.relu(self.conv(x_padded)) + x

# Attention block
class SelfAttentionBlock(nn.Module):
    """Pre-LN self-attention block: LayerNorm -> MHA -> residual -> LayerNorm -> FFN -> residual"""
    
    def __init__(self, embed_dim, num_heads=4, ffn_dim=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x):
        # x: (T, batch, embed_dim
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]
        x = x + self.ffn(self.norm2(x))
        return x

# Full model: DeclipNet
class DeclipNet(nn.Module):
    """
    U-Net with dilated conv bottleneck and self-attention for speech declipping.
    
    Architecture:
        - Encoder: 4 blocks, H=8 base channels, K=8, S=2
        - Dilated conv stack: 3 blocks, dilation 1/2/4, with residuals
        - Bottleneck: N self-attention blocks, embed_dim=64, num_heads=4
        - Decoder: 4 blocks symmetric with encoder, skip connections from encoder
    
    Args:
        H: base channel count (default 8)
        N: number of self-attention blocks (default 3)
        num_heads: attention heads (default 4)
        ffn_dim: feed-forward dimension in attention blocks (default 256)
    """

    def __init__(self, H=8, N=3, num_heads=4, ffn_dim=256):
        super().__init__()

        # Encoder
        self.encoders = nn.ModuleList([
            EncoderBlock(1,       H,     ),
            EncoderBlock(H,       H*2,   ),
            EncoderBlock(H*2,     H*4,   ),
            EncoderBlock(H*4,     H*8,   ),
        ])

        # Dilated conv stack
        self.dilated = nn.ModuleList([
            DilatedBlock(H*8, dilation=1),
            DilatedBlock(H*8, dilation=2),
            DilatedBlock(H*8, dilation=4),
        ])

        # Self-attention blocks
        embed_dim = H * 8
        self.attention = nn.ModuleList([
            SelfAttentionBlock(embed_dim, num_heads=num_heads, ffn_dim=ffn_dim)
            for _ in range(N)
        ])

        # Decoder
        self.decoders = nn.ModuleList([
            DecoderBlock(H*8, H*4,  ),
            DecoderBlock(H*4, H*2,  ),
            DecoderBlock(H*2, H,    ),
            DecoderBlock(H,   1, last=True),
        ])

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            
    def forward(self, x):
        # Encoder + store skips
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Dilated conv stack
        for dilated in self.dilated:
            x = dilated(x)

        # Self-attention
        # (batch, C, T) -> (T, batch, C)
        x = x.permute(2, 0, 1)
        for attn in self.attention:
            x = attn(x)
        # (T, batch, C) -> (batch, C, T)
        x = x.permute(1, 2, 0)

        # Decoder + skip connections in reverse order
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return x