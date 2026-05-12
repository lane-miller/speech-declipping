# 02_deployment — Export, Optimization & Benchmarking

ONNX and Core ML export of DeclipNet, post-training optimization attempts, and latency/quality benchmarking across runtimes and precisions.

## `deploy_util.py`

- `eval_ola_routed`: OLA+routed test evaluation — 50%-overlap Hanning-windowed blocks, f_c-based bypass routing, returns mean SI-SDR / PESQ across utterances
- Accepts any callable model (PyTorch, ONNX wrapper, CoreML wrapper)

## `02_pruning.ipynb`

- Structured channel pruning on all Conv1d layers at sparsity [20%, 40%, 60%, 80%]
- Quality collapsed at all levels — SI-SDR dropped 3–10 dB even at 20% sparsity
- Post-prune fine-tuning did not recover quality
- Conclusion: pruning not viable for this architecture at H=8

## `02_export.ipynb`

- ONNX export: opset 17, dynamic batch dim, 452-node graph; max diff vs PyTorch 5.72e-06 (float32 exact)
- Core ML export: `mlprogram` format, `compute_units=ALL`, macOS 13+; max diff vs PyTorch 2.20e-02 (expected from fp16 ANE pipeline)
- Saves `models/declipnet.onnx` and `models/declipnet.mlpackage`

## `02_benchmark.ipynb`

- INT8 post-training quantization via `linear_quantize_weights` — 41% size reduction (691 → 408 KB), negligible quality/latency change
- Per-block latency: PyTorch MPS ~2.2 ms, ONNX CPU ~0.66 ms, CoreML FP32/INT8 ~0.48 ms
- OLA test set (100 utterances): all variants within 0.08 dB SI-SDR and 0.007 PESQ
- FP32 vs INT8 CoreML is a wash at 314k params — weights fit in cache, no bandwidth gain from quantization
- Recommendation: CoreML FP32 `.mlpackage` for deployment
