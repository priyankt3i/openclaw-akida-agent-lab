# Prototype-1 Experiment Report

Date: 2026-04-07 UTC

## Summary

Current best surrogate result:
- Threshold: `0.20`
- Sparsity: `0.8671875`
- Kept fraction: `0.1328125`
- Efficiency gain proxy: `7.529411764705882x`
- Relative MSE: `0.0026608324620881955` (~0.266%)

This means the prototype surrogate has already exceeded the 5x efficiency target while remaining under the 1% quality-loss budget. The main remaining risk is not the sparse math, but whether this win survives an Akida-compatible implementation path.

## Experiments Completed

### 1. Prototype scaffold
- Created `prototype-1/` with `experiments/`, `src/`, and `artifacts/`.
- Purpose: isolate spiking-attention surrogate experiments without fragmenting the repository.

### 2. Dense baseline
Script:
- `prototype-1/experiments/baseline_attention.py`

Result:
- MSE: `0.00014356268078280677`
- Relative L2 error: `0.029943797528364076`
- Attention mean abs delta: `0.003306078488015297`
- Activation sparsity: `0.142578125`
- Efficiency gain proxy: `1.128747795414462`

Takeaway:
- Baseline harness works, but the initial sparse behavior was too conservative to approach the project target.

### 3. Sparse threshold attention
Script:
- `prototype-1/experiments/sparse_attention.py`

Result:
- Kept fraction: `0.140625`
- Energy proxy: `0.140625`
- Relative MSE: `0.0005401872214067399`

Takeaway:
- Strong pruning with low error, enough to justify a broader threshold sweep.

### 4. Threshold sweep
Script:
- `prototype-1/experiments/threshold_sweep.py`

Artifacts:
- `prototype-1/artifacts/threshold_sweep_results.json`
- `prototype-1/artifacts/threshold_sweep_summary.md`

Best operating point:
- Threshold: `0.20`
- Sparsity: `0.8671875`
- Kept fraction: `0.1328125`
- Efficiency gain proxy: `7.529411764705882x`
- Relative MSE: `0.0026608324620881955`

More aggressive point:
- Threshold `0.50`
- Efficiency gain proxy: `8.0x`
- Relative MSE: `0.020431983488056943`

Takeaway:
- Threshold `0.20` is the best current reproducible frontier point.
- Several threshold settings collapse to the same mask pattern on this tiny deterministic setup, so the tradeoff curve is quantized rather than smooth.

### 5. Top-k comparison
Script:
- `prototype-1/experiments/topk_attention_compare.py`

Artifact:
- `prototype-1/artifacts/topk_compare_metrics.json`

Threshold gating (`threshold=0.15`):
- Kept fraction: `0.140625`
- Efficiency gain proxy: `7.11x`
- Relative MSE: `5.40e-4`

Top-k gating (`k=2`):
- Kept fraction: `0.25`
- Efficiency gain proxy: `4.0x`
- Relative MSE: `3.32e-5`

Takeaway:
- Top-k preserves quality better but is too dense at fixed `k=2` to meet the current efficiency target.
- Threshold gating is stronger for our objective.

### 6. Spike/event proxy
Script:
- `prototype-1/experiments/spike_proxy_attention.py`

Artifacts:
- `prototype-1/artifacts/spike_proxy_metrics.json`
- `prototype-1/artifacts/spike_proxy_sweep.csv`
- `prototype-1/artifacts/spike_proxy_traces.npz`

Best sub-1% loss point:
- Delta threshold: `0.1`
- Row spike threshold: `1`
- Top-k: `3`
- Quality loss: `0.8089862374620973%`
- Attention kept fraction: `0.375`
- Sparsity: `0.625`
- Energy gain proxy: `2.6666666666666665x`

Takeaway:
- Interesting event-driven behavior, but weaker than threshold gating on efficiency.

### 7. Lava feasibility
Artifact:
- `prototype-1/artifacts/lava_feasibility.md`

Result:
- No local Lava runtime installed.
- Docs are reachable.
- Sparse surrogate appears conceptually portable into Lava-style components.
- Exact softmax attention remains a poor fit for spike-native execution.

Takeaway:
- Lava is plausible for research prototyping, but not yet runnable here.

### 8. Meta-TF / Akida feasibility
Artifact:
- `prototype-1/artifacts/metatf_akida_feasibility.md`

Result:
- `akida`, `cnn2snn`, `tensorflow`, and `tf_keras` import successfully.
- No native attention primitive is obvious.
- Direct `MultiHeadAttention` conversion appears blocked.
- `akida_models` import is blocked by missing `libGL.so.1`.
- Even tiny dense models need careful input/quantization handling.

Takeaway:
- The viable path is an Akida-friendly surrogate built from supported primitives, not direct softmax/MHA conversion.

## How We Improved

We moved from a baseline efficiency proxy of roughly `1.13x` to a reproducible threshold-gated surrogate result of `7.53x`, while keeping relative MSE to about `0.266%`.

This is a substantial algorithmic improvement. The project question has shifted from:
- "Can sparse/spiking attention-like behavior work at all?"

to:
- "Can we preserve that win in an Akida-compatible implementation?"

## Current Decision

- Algorithmic direction: promising
- Toolchain conversion: still unproven
- Hardware purchase: not recommended yet

## Next Step

Build a tiny `tf_keras` quantized gating surrogate from Akida-supported primitives, then test:
- `cnn2snn.check_model_compatibility`
- quantization
- conversion
