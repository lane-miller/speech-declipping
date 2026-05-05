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