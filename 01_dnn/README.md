# 01_dnn — DNN-Based Speech Declipping

Waveform-domain U-Net trained to restore clipped speech segments. Includes the model, loss functions, training utilities, and notebooks for validation, ablation, and final training.

## `util.py`

- `DeclipDataset`: loads pre-chunked train/val blocks, applies clipping on-the-fly from manifest alpha
- `EncoderBlock`, `DecoderBlock`, `DilatedBlock`, `SelfAttentionBlock`: submodules of `DeclipNet`
- `DeclipNet`: 1D waveform-domain U-Net — H=8 base channels, D=4 encoder/decoder layers (K=8, S=2), dilated conv bottleneck (d=1,2,4), N=3 Pre-LN self-attention blocks (embed_dim=64, num_heads=4), skip connections follow CleanUNet convention, ~314k parameters
- `l1_loss`, `weighted_l1_loss` (amplitude-weighted, `weight_power` param), `multires_stft_loss` (scaled by 1/20, `weight_power` param), `dwt_loss` (scaled by 0.5, `weight_power` param, differentiable via `pytorch_wavelets`)
- `si_sdr`: per-sample scale-invariant SDR metric
- `train_run`: full training loop with early stopping, optional LR scheduling, optional `torch.compile`, optional gradient clipping, `val_every` param, saves best model checkpoint and results JSON

## `01_dataset.ipynb`

- Instantiates `DeclipDataset` and `DataLoader` for train and val
- Verifies batch shapes (32, 1, 1024), clipping correctly applied, no silent blocks

## `01_model.ipynb`

- Instantiates `DeclipNet`, verifies forward pass shape (batch, 1, 1024) → (batch, 1, 1024)
- Confirms 314k trainable parameters
- Verifies encoder skip shapes via forward hooks
- Confirms output is unconstrained (no output activation)
- Confirms std does not collapse across decoder layers with Kaiming init

## `01_loss.ipynb`

- Unit tests all loss functions with physically motivated inputs
- Verifies amplitude weighting emphasis on clipped regions
- Verifies STFT and DWT frequency weighting increases loss for low-pass filtered inputs
- Confirms gradient flow through all differentiable loss functions

## `01_train_study.ipynb`

- Loss function ablation study, sequential runs with early stopping (patience=10)
- Stage 1a: L1 vs weighted L1 — weighted L1 wins (21.25 vs 18.67 dB val SI-SDR)
- Stage 1b: `weight_power` tuning p=1,2,4 — p=1 wins marginally, faster convergence
- Stage 1c: weighted L1 + STFT vs weighted L1 + DWT — DWT wins marginally (21.93 vs 21.90 dB), faster convergence; STFT scaled by 1/20, DWT scaled by 0.5 to balance loss magnitudes
- Winning loss: `weighted_l1 + dwt_loss`

## `01_train_final.ipynb`

- Full training run with winning loss: `weighted_l1 + dwt_loss`
- Optimizations: `torch.compile`, gradient clipping (considered but not implemented in final run), `ReduceLROnPlateau` LR scheduling, val_every=1, patience=15
- Converged in 162 epochs, best val SI-SDR: 22.43 dB

## `01_train_results.ipynb`

- Loads and plots train loss, val SI-SDR, and LR schedule for final training run
- Computes clipped input SI-SDR baseline on val set: 15.75 dB
- Val SI-SDR improvement over clipped baseline: 6.68 dB
- Test set evaluation (100 utterances, 20 held-out talkers):
  - Without routing: SI-SDR 25.37 dB (+4.30 over baseline), PESQ 3.63 (+0.31)
  - With routing (FC_BYPASS=0.015, 78.4% blocks bypassed): SI-SDR 27.02 dB (+5.94), PESQ 3.85 (+0.53)
  - With routing + 50% overlap-add (Hanning window): SI-SDR 27.32 dB (+6.25), PESQ 3.95 (+0.62) — +0.31 dB SI-SDR, +0.09 PESQ over non-overlapping
- Stratified by alpha (OLA+routed): largest gains at low alpha (severe clipping) — PESQ +1.18, SI-SDR +7.82 dB
- High alpha utterances (lightly clipped) mostly bypassed correctly; small gains reflect ceiling effect


## Limitations and Future Work

- **Model capacity**: H=8 (~314k parameters) chosen for local MPS training feasibility; scaling H
  would likely improve performance
- **Hyperparameter tuning**: learning rate, batch size, H, N_ATTN, NUM_HEADS not searched —
  default values used throughout; a proper grid or Bayesian search could yield meaningful gains
- **Ablation study**: loss function runs were relatively short and subject to variance; margins
  between some configurations were small enough that results should be interpreted directionally
- **Training data**: ~4 hours of LibriSpeech train-clean-100; larger corpus coverage (e.g. full
  100 hours) would likely improve generalization, particularly across speaking styles and recording
  conditions
- **PESQ during training**: val SI-SDR used as proxy metric due to pre-chunked dataset; utterance-level
  PESQ tracked only at final test evaluation — periodic PESQ on held-out utterances during training
  would give a more perceptually grounded early stopping signal
- **Routing threshold**: FC_BYPASS=0.015 set empirically from 00_routing_analysis; systematic
  search over threshold values on the test set could optimize the bypass/process tradeoff
- **No light/heavy specialist split**: single model trained on full clipping severity range;
  specialist networks with severity-stratified training data could improve performance at extremes
- **Deployment**: ONNX export, CoreML conversion, and on-device inference on Apple A15 identified
  as natural next steps; quantization comparison (INT8 vs FP32) on latency and SI-SDR is planned