# Akida surrogate probe, 2026-04-08 UTC

## What was tested
- `prototype-1/experiments/akida_surrogate_probe.py`
- A minimal `tf_keras` ReLU6-gated surrogate with only `Dense` layers
- A small `Conv2D + DepthwiseConv2D + ReLU6 + Conv2D` probe to test the depthwise-style path

## Proven in this environment
- A `tf_keras.Sequential` model shaped as `Dense -> ReLU6 -> Dense -> ReLU6 -> Dense` can be quantized with `cnn2snn.quantize(...)` and converted with `cnn2snn.convert(...)`.
- The successful path requires signed input assumptions. `check_model_compatibility(..., input_dtype='int8')` passes, while the same dense surrogate is blocked for `uint8` because of input-sign incompatibility.
- The converted model reports: `akida.Model, layer_count=4, sequence_count=1, output_shape=[1, 1, 16]`.
- Measured activation sparsity on deterministic synthetic input was about `49.0%` after the first ReLU6 gate and `50.8%` after the second ReLU6 gate.

## Blocked / not yet proven
- The depthwise-style probe passes the light compatibility check, but the real conversion path still fails after quantization.
- Current failure: `Layer mix_dw3x3 of type DepthwiseConv2D is not supported for Akida conversion`.
- So, in this setup, `DepthwiseConv2D` should be treated as a probe only, not a validated deployment primitive.

## Takeaway
The cleanest Akida-friendly surrogate we have actually proven here is a signed-int8, ReLU6-gated dense token mixer, not yet a conv-depthwise hybrid.
