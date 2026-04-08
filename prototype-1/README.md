# prototype-1

Current sandbox for the Akida attention-surrogate work.

## Status
- The manual history-buffer branch is retained as evidence, not as a product path.
- The active engineering pivot is an Akida-native temporal path built around `DepthwiseBufferTempConv` and related quantized/conversion gates.
- The question now is not whether sparse gating works in software, but whether a temporal surrogate can survive the real Akida toolchain and keep token-position error controlled after warmup.

## Active files
Engineering work that reflects the current direction lives mainly in:
- `experiments/depthwise_buffer_tempconv_probe.py`
- `experiments/conversion_gate_depthwise_buffer_tempconv_min.py`
- `experiments/conversion_gate_qdepthwisebuffer_tempconv_exact.py`
- `experiments/signed_impulse_depthwise_buffer_tempconv_min.py`
- `experiments/temporal_path_eval_contract.py`
- `artifacts/coordination_status.json`
- `artifacts/conversion_gate_qdepthwisebuffer_tempconv_exact.json`
- `artifacts/depthwise_buffer_tempconv_probe.json`
- `artifacts/signed_impulse_depthwise_buffer_tempconv_min.json`

## Archived or superseded material
These files remain useful for diagnosis and historical comparison, but they are not the forward product direction. They now live under `archive/`:
- `archive/experiments/history_buffer/`
- `archive/artifacts/history_buffer/`
- `archive/artifacts/toy_baselines/`
- `archive/artifacts/stakeholder/`
- `archive/artifacts/reports/`

## Working interpretation
- Software-side sparse attention still looks promising.
- Manual buffer repair did not become a credible deployment path.
- `DepthwiseBufferTempConv` is the strongest Akida-native clue found so far, even though the exact quantized/conversion path still needs validation.

## Practical repo map
- `experiments/`: runnable probes and validation scripts
- `artifacts/`: captured metrics, reports, and conversion diagnostics
- `dashboard.py`: lightweight status view across the prototype outputs

If you are picking work back up, start with `artifacts/coordination_status.json`, then the `DepthwiseBufferTempConv` probe and conversion-gate scripts, not the manual history-buffer branch.
