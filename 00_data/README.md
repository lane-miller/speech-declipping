# 00_data — Dataset Preparation
Scripts and notebooks for building, verifying, and analyzing the speech-declipping dataset derived from LibriSpeech train-clean-100


- **`config.py`** — Shared constants (sample rate, block size, clipping thresholds, silence gate, split durations) and dataset paths used by all notebooks in this folder  


- **`00_build_dataset.ipynb`** — Builds full dataset end-to-end:
  - Partitions 20 held-out test talkers (10 M / 10 F); saves full-length normalized test utterances with α in the manifest for on-the-fly clipping at evaluation time
  - Generates pre-chunked train/val block tensors (`train_blocks.pt`, `val_blocks.pt`) with per-block talker ID and clipping threshold α stored in manifests  


- **`00_verify_dataset.ipynb`** — Checks on the generated tensors: 
   - shape verification
   - manifest length consistency
   - α-distribution uniformity
   - talker-distribution uniformity
   - silence-filter validation  


- **`00_routing_analysis.ipynb`** — Empirically determines bypass threshold ε by sampling non-test utterances across a grid of clipping severities and measuring f_c (fraction clipped), PESQ MOS as function of f_c