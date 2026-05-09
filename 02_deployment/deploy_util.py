"""
util.py
-------
Shared utilities for 02_deployment notebooks.
"""

import sys
import json
import torch
import numpy as np
from pesq import pesq as pesq_fn

sys.path.insert(0, "../01_dnn")
sys.path.insert(0, "..")
from config import *
from util import si_sdr


def eval_ola_routed(model, test_manifest, device,
                    fc_bypass=0.015, hop=BS // 2, verbose=True):
    """
    OLA+routed test evaluation. Returns mean SI-SDR and PESQ across utterances.

    For each utterance: extracts 50%-overlapping blocks, routes lightly-clipped
    blocks (f_c <= fc_bypass) as bypass, runs model on the rest, then
    reconstructs via Hanning-windowed overlap-add.

    Args:
        model:         DeclipNet in eval mode
        test_manifest: list of dicts with 'file_idx' and 'alpha'
        device:        torch device
        fc_bypass:     clipped-fraction threshold below which blocks are bypassed
        hop:           hop size for OLA (default BS//2 = 50% overlap)
        verbose:       print per-utterance results

    Returns:
        dict with keys: mean_sdr, mean_pesq, mean_baseline_sdr, mean_baseline_pesq,
                        sdr_list, pesq_list, baseline_sdr_list, baseline_pesq_list,
                        blocks_bypassed, blocks_processed
    """
    window = torch.hann_window(BS, device=device)

    sdr_list, pesq_list = [], []
    baseline_sdr_list, baseline_pesq_list = [], []
    blocks_bypassed = 0
    blocks_processed = 0

    for entry in test_manifest:
        clean = torch.load(
            TEST_OUT / f"test_{entry['file_idx']:03d}_clean.pt",
            weights_only=True
        ).to(device)

        peak = clean.abs().max()
        threshold = entry["alpha"] * peak
        clipped = clean.clamp(-threshold, threshold)

        sig_len = clean.shape[1]
        if sig_len < BS:
            continue

        n_blocks = (sig_len - BS) // hop + 1
        trimmed_len = (n_blocks - 1) * hop + BS

        clipped_chunks = (
            clipped[:, :trimmed_len]
            .unfold(1, BS, hop)
            .squeeze(0)
            .unsqueeze(1)
        )

        fc_per_block = (
            (clipped_chunks.abs() >= threshold * 0.9999)
            .float().mean(dim=-1).squeeze(1)
        )
        needs_processing = fc_per_block > fc_bypass

        blocks_bypassed += (~needs_processing).sum().item()
        blocks_processed += needs_processing.sum().item()

        output_chunks = clipped_chunks.clone()
        if needs_processing.any():
            with torch.no_grad():
                output_chunks[needs_processing] = model(clipped_chunks[needs_processing])

        output_signal = torch.zeros(1, trimmed_len, device=device)
        window_sum = torch.zeros(1, trimmed_len, device=device)

        for i in range(n_blocks):
            start = i * hop
            output_signal[:, start:start + BS] += output_chunks[i, 0, :] * window
            window_sum[:, start:start + BS] += window

        output_signal = output_signal / window_sum.clamp(min=1e-8)

        clean_trimmed = clean[:, :trimmed_len]
        clipped_trimmed = clipped[:, :trimmed_len]

        utt_sdr = si_sdr(clean_trimmed, output_signal).mean().item()
        baseline_sdr = si_sdr(clean_trimmed, clipped_trimmed).mean().item()
        sdr_list.append(utt_sdr)
        baseline_sdr_list.append(baseline_sdr)

        clean_np = clean_trimmed.squeeze().cpu().numpy()
        output_np = output_signal.squeeze().cpu().numpy()
        clipped_np = clipped_trimmed.squeeze().cpu().numpy()

        try:
            utt_pesq = pesq_fn(FS, clean_np, output_np, "wb")
            baseline_pesq = pesq_fn(FS, clean_np, clipped_np, "wb")
        except Exception as e:
            if verbose:
                print(f"PESQ failed for utt {entry['file_idx']:03d}: {e}")
            utt_pesq = float("nan")
            baseline_pesq = float("nan")

        pesq_list.append(utt_pesq)
        baseline_pesq_list.append(baseline_pesq)

        if verbose:
            print(
                f"utt {entry['file_idx']:03d} | "
                f"SDR: {baseline_sdr:.2f} -> {utt_sdr:.2f} dB "
                f"(delta: {utt_sdr - baseline_sdr:.2f}) | "
                f"PESQ: {baseline_pesq:.3f} -> {utt_pesq:.3f}",
                flush=True,
            )

    total_blocks = blocks_bypassed + blocks_processed
    mean_sdr = np.mean(sdr_list)
    mean_pesq = np.nanmean(pesq_list)
    mean_bl_sdr = np.mean(baseline_sdr_list)
    mean_bl_pesq = np.nanmean(baseline_pesq_list)

    if verbose:
        print(f"\nRouting: {blocks_bypassed}/{total_blocks} blocks bypassed "
              f"({100 * blocks_bypassed / total_blocks:.1f}%)")
        print(f"\nMean SI-SDR (OLA+routed):  {mean_sdr:.4f} dB")
        print(f"Mean baseline SI-SDR:      {mean_bl_sdr:.4f} dB")
        print(f"SI-SDR improvement:        {mean_sdr - mean_bl_sdr:.4f} dB")
        print(f"\nMean PESQ (OLA+routed):    {mean_pesq:.4f}")
        print(f"Mean baseline PESQ:        {mean_bl_pesq:.4f}")
        print(f"PESQ improvement:          {mean_pesq - mean_bl_pesq:.4f}")

    return {
        "mean_sdr": mean_sdr,
        "mean_pesq": mean_pesq,
        "mean_baseline_sdr": mean_bl_sdr,
        "mean_baseline_pesq": mean_bl_pesq,
        "sdr_list": sdr_list,
        "pesq_list": pesq_list,
        "baseline_sdr_list": baseline_sdr_list,
        "baseline_pesq_list": baseline_pesq_list,
        "blocks_bypassed": blocks_bypassed,
        "blocks_processed": blocks_processed,
    }
