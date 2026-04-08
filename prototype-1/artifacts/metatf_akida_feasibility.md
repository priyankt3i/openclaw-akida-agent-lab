# Meta-TF / Akida feasibility note

Date: 2026-04-07 21:10:36 UTC

## Bottom line
Partial feasibility. The current prototype direction is promising only if we treat the sparse/spiking attention block as an Akida-friendly surrogate, not as native transformer attention.

## What is compatible
- Local stack is present: `akida 2.19.1`, `cnn2snn 2.19.1`, `tensorflow 2.19.1`, `tf_keras 2.19.0`.
- Akida exposes low-level building blocks such as `Model`, `InputData`, `Conv2D`, `Dense1D`, and quantization layers.
- The current surrogate already shows strong intrinsic sparsity: baseline attention weights below threshold = `0.859375`, kept fraction after masking = `0.140625`, energy proxy = `0.140625`, implying about `7.1x` active-weight reduction versus dense.
- This suggests the project goal may be reachable if attention is re-expressed as quantized gating, sparse selection, or MLP/conv-style primitives that Akida can map.

## What looks blocked
- No obvious transformer/attention primitive exists in the exposed `akida` or `cnn2snn` API surface.
- `cnn2snn.check_model_compatibility` rejects a `tf_keras` `MultiHeadAttention` model with: `Impossible to quantize 'model'. At least one layer should be quantizable.`
- Using `keras` 3 functional models also fails earlier because `cnn2snn` expects `tf_keras` model classes, so Keras frontend/version mismatch is another practical blocker.
- `akida_models` import currently fails with `ImportError: libGL.so.1: cannot open shared object file`, so reference-model exploration is incomplete until the container adds that shared library.
- Even a tiny `tf_keras` dense model did not pass end-to-end compatibility as written, hitting `InputData sign incompatible with previous layer`, so Akida conversion will need careful input/quantization conventions, not just layer substitution.

## Assessment for current sparse/spiking attention path
- Compatible direction: keep the repo's current thresholded sparse attention experiment as a software reference and distill it into Akida-supported quantized operations.
- Likely compatible pieces: tokenwise projections, low-bit gating, top-k or threshold masks approximated by quantized activations, event/sparsity accounting, and possibly per-token processing via dense/conv primitives.
- Likely incompatible piece: direct softmax attention / native MHA conversion.

## Recommended next step
Build a tiny `tf_keras` surrogate that replaces softmax attention with Akida-friendly quantized gating and verify `check_model_compatibility -> quantize -> convert` on that surrogate before investing in a fuller Meta-TF style block.
