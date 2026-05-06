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
        self.convtranspose = nn.ConvTranspose1d(in_channels, out_channels,
                                                kernel_size, stride=stride,
                                                padding=padding, output_padding=0)
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
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x):
        # x: (batch, T, embed_dim)
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
        # (batch, C, T) -> (batch, T, C)
        x = x.transpose(1, 2)
        for attn in self.attention:
            x = attn(x)
        # (batch, T, C) -> (batch, C, T)
        x = x.transpose(1, 2)

        # Decoder + skip connections in reverse order
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return x


# =============================================================================
# Loss
# =============================================================================

def l1_loss(output, target):
    """Standard L1 loss on raw waveform."""
    return F.l1_loss(output, target)


def weighted_l1_loss(output, target, clipped, weight_power=1.0):
    """
    Amplitude-weighted L1 loss. Weights each sample's error by its normalized
    absolute amplitude in the clipped input raised to weight_power.
    
    weight_power=0: uniform weighting (equivalent to standard L1)
    weight_power=1: linear emphasis on high amplitude samples
    weight_power>1: increasingly aggressive emphasis on clipping boundary
    
    Args:
        output:       model output (batch, 1, BS)
        target:       clean reference (batch, 1, BS)
        clipped:      clipped input (batch, 1, BS)
        weight_power: exponent for amplitude weighting (default 1.0)
    """
    peak = clipped.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    weights = (clipped.abs() / peak).pow(weight_power)
    return (weights * (output - target).abs()).mean()

def multires_stft_loss(output, target, fft_sizes=(64, 256, 512),
                       hop_sizes=(16, 64, 128), win_sizes=(64, 256, 512),
                       weight_power=1.0):
    """
    Multi-resolution STFT loss. L1 on magnitude spectrograms at multiple
    resolutions, with optional frequency-dependent weighting.

    Args:
        output:            model output (batch, 1, BS)
        target:            clean reference (batch, 1, BS)
        fft_sizes:         FFT bin counts per resolution
        hop_sizes:         hop sizes per resolution
        win_sizes:         window lengths per resolution
        weight_power: exponent for frequency-dependent weighting.
                           1.0 = linear emphasis on high freqs,
                           0.0 = no weighting (uniform)
    """
    output = output.squeeze(1)  # (batch, BS)
    target = target.squeeze(1)

    total_loss = 0.0

    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        # Compute magnitude spectrograms
        window = torch.hann_window(win_size, device=output.device)
        out_stft = torch.stft(output, fft_size, hop_size, win_size,
                      window=window, center=False, return_complex=True)
        tgt_stft = torch.stft(target, fft_size, hop_size, win_size,
                      window=window, center=False, return_complex=True)

        out_mag = out_stft.abs()  # (batch, freq_bins, time_frames)
        tgt_mag = tgt_stft.abs()

        # Frequency-dependent weights: higher freqs weighted more
        n_freqs = out_mag.shape[1]
        freq_weights = torch.linspace(1 / n_freqs, 1, n_freqs, device=output.device) \
                   .pow(weight_power)
        freq_weights = freq_weights / freq_weights.mean()  # normalize so mean weight = 1
        freq_weights = freq_weights.view(1, -1, 1)

        loss = (freq_weights * (out_mag - tgt_mag).abs()).mean()
        total_loss += loss

    return total_loss / len(fft_sizes)


def dwt_loss(output, target, wavelet='db4', levels=4, weight_power=1.0):
    """
    Discrete wavelet transform loss. L1 on wavelet subband coefficients
    at multiple decomposition levels, with optional frequency-dependent weighting.
    Higher subband indices correspond to higher frequency detail coefficients.
    Subband weights are normalized so mean weight = 1, ensuring weight_power=0
    gives identical result to uniform weighting.

    Args:
        output:       model output (batch, 1, BS)
        target:       clean reference (batch, 1, BS)
        wavelet:      wavelet name (default 'db4')
        levels:       number of decomposition levels (default 4)
        weight_power: exponent for subband weighting.
                      0.0 = uniform, 1.0 = linear high freq emphasis,
                      >1.0 = increasingly aggressive high freq emphasis
    """
    from pytorch_wavelets import DWT1DForward

    dwt = DWT1DForward(wave=wavelet, J=levels, mode='periodization')
    dwt = dwt.to(output.device)

    out_yl, out_yh = dwt(output)
    tgt_yl, tgt_yh = dwt(target)

    # Order subbands from lowest to highest frequency:
    # approximation, then details from coarsest to finest
    out_subbands = [out_yl] + list(reversed(out_yh))
    tgt_subbands = [tgt_yl] + list(reversed(tgt_yh))
    n_subbands = len(out_subbands)

    # Normalized subband weights: index 0 = lowest freq, index N-1 = highest freq
    weights = torch.linspace(1 / n_subbands, 1, n_subbands, device=output.device, dtype=output.dtype)
    weights = weights.pow(weight_power)
    weights = weights / weights.mean()

    total_loss = sum(
        weights[i] * F.l1_loss(oc, tc)
        for i, (oc, tc) in enumerate(zip(out_subbands, tgt_subbands))
    ) / n_subbands

    return total_loss


# =============================================================================
# Metrics
# =============================================================================

def si_sdr(reference, degraded):
    """
    Scale-invariant signal-to-distortion ratio.
    Higher is better. Returns per-example values in dB.

    Args:
        reference: clean reference tensor (batch, 1, BS) or (1, BS)
        degraded:  degraded/output tensor, same shape as reference

    Returns:
        Tensor of shape (batch,) with SI-SDR in dB for each example.
    """
    reference = reference - reference.mean(dim=-1, keepdim=True)
    degraded = degraded - degraded.mean(dim=-1, keepdim=True)
    dot = (degraded * reference).sum(dim=-1, keepdim=True)
    s_ref = (reference * reference).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    alpha = dot / s_ref
    signal = alpha * reference
    noise = degraded - signal
    return 10 * torch.log10(
        (signal ** 2).sum(dim=-1) / (noise ** 2).sum(dim=-1).clamp(min=1e-8)
    ).squeeze(-1)