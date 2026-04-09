# Reliable Conversion Path for Autoregressive LLMs on Spiking Hardware

## Draft status

Working manuscript draft, aligned to the current state of `prototype-1` as of 2026-04-08.
This draft is intentionally publication-shaped but technically conservative. It centers the claim on the **conversion path** and the **diagnostic methodology**, not on a final product-quality LLM result.

## Title

**Reliable Conversion Path for Autoregressive LLMs on Spiking Hardware**

## Authors

Priyank et al.  
OpenClaw Akida Agent Lab

## Abstract

Autoregressive large language models (LLMs) remain difficult to deploy on neuromorphic hardware because standard attention-style temporal mixing does not survive direct conversion into event-driven execution. We present a practical conversion study on BrainChip Akida 2.0 tooling that identifies failure modes in manual history-buffer surrogates and establishes a hardware-native temporal route based on `DepthwiseBufferTempConv`. Our experiments show that a manually packed signed history buffer can appear numerically stable while still corrupting inhibitory temporal structure at the hardware boundary, yielding a "stable but blurry" approximation. In contrast, an Akida-native temporal primitive preserves signed inhibitory behavior under conversion, and we demonstrate clean fused conversion for a kernel-4 temporal unit and for a stacked two-layer bottlenecked temporal block. These results define a reliable structural conversion path for autoregressive temporal mixing on spiking hardware, even though end-to-end task fidelity remains an open problem. We contribute (1) a reproducible diagnostic workflow for identifying signed-history corruption and conversion-pattern mismatches, (2) evidence that Akida-native temporal FIFO primitives preserve inhibitory temporal semantics that manual buffers lose, and (3) a minimal fused stacked temporal cell that can serve as the base unit for future autoregressive spiking LLM architectures.

## 1. Introduction

Neuromorphic deployment of autoregressive language models is attractive because event-driven temporal processing offers a plausible route to large energy savings relative to dense matrix-multiplication-dominated inference. However, the standard transformer recipe assumes unrestricted signed activations, large temporal context windows, and exact high-level software semantics. These assumptions are poorly aligned with present-day spiking hardware conversion paths.

The central engineering question is not whether spiking language modeling is conceptually possible, but whether a **reliable conversion path** exists from an autoregressive temporal surrogate to real neuromorphic execution primitives. In practice, many candidate designs fail before reaching hardware, or convert in ways that silently destroy the inhibitory temporal information needed for coherent sequence processing.

This work studies that conversion problem directly on Akida 2.0 tooling. We began with a signed-int8 Dense/ReLU6 history-buffer surrogate designed to approximate a causal temporal mixer. That path improved drift relative to a trivial current-only baseline, but it failed structurally at the Akida boundary: negative history values were partially erased or aliased when packed into the hardware input contract. This failure produced a model that was more stable than expected over long context lengths, but too lossy for product-quality use.

Rather than continuing to rescue the manual buffer, we pivoted to Akida-native temporal primitives. The key breakthrough was confirming that `DepthwiseBufferTempConv` preserves signed inhibitory temporal behavior under conversion and that a fused stacked temporal bottleneck block can be represented cleanly in the Akida model graph. This does not yet solve final model fidelity, but it does solve the more basic and often neglected problem of finding a structural conversion path that the silicon actually supports.

## 2. Contributions

This draft claims three concrete contributions.

1. **Failure analysis of manual history-buffer surrogates.**  
   We show that manual signed history packing can fail at the hardware boundary even when float-domain drift metrics appear acceptable.

2. **A reliable structural conversion path.**  
   We identify `DepthwiseBufferTempConv` as an Akida-native temporal primitive that preserves inhibitory temporal behavior under conversion and supports at least one clean fused operating point (`kernel_size = 4`), including a stacked two-layer bottleneck block.

3. **A diagnostic methodology for neuromorphic autoregressive conversion.**  
   We provide a practical sequence of tests: weight saturation audit, activation audit, long-context drift sweep, signed-basis impulse tests, conversion-gate probes, and per-kernel fusion mapping.

## 3. Related Context

Recent work on low-bit or matmul-free language modeling, including ternary-weight and event-driven directions, suggests that the dominant compute patterns of LLMs can be rethought. However, many such results are either algorithmic proofs of concept or depend on custom hardware or FPGA deployments. Our focus is complementary: not on proving that low-power temporal language computation is theoretically attractive, but on establishing a **reproducible conversion path on commercially targeted spiking hardware tooling**.

## 4. Problem Statement

We study the following problem:

> Can an autoregressive temporal surrogate be converted into a spiking-hardware execution path without losing the signed inhibitory temporal information required for coherent sequence processing?

A useful answer must satisfy more than float-domain accuracy. It must also survive:

- `cnn2snn` compatibility checks,
- quantization,
- conversion into an Akida model,
- preservation of inhibitory temporal behavior,
- and eventual composability into deeper temporal stacks.

## 5. Experimental Setup

### 5.1 Manual history-buffer surrogate

The initial surrogate used a signed-int8-friendly Dense/ReLU6 temporal mixer over a small causal history window. This path was chosen because it matched supported primitive families and could be checked through `cnn2snn` compatibility, quantization, and conversion.

### 5.2 Evaluation metrics

We tracked:

- global relative MSE,
- relative L2 error,
- token-position error curves,
- late-to-early error ratio,
- activation sparsity,
- activation saturation and variance trends,
- quantized-vs-float delta,
- and hardware-conversion success/failure.

### 5.3 Conversion gates

We used a staged gate sequence:

1. **Compatibility gate:** `check_model_compatibility(...)`
2. **Quantization gate:** successful creation of quantized layers
3. **Conversion gate:** successful `cnn2snn.convert(...)`
4. **Behavior gate:** signed impulse preservation and fused-layer inspection

## 6. Results

### 6.1 Manual history buffer improved stability but remained too lossy

A 4-token repaired history-buffer path materially improved over a current-only baseline and remained compatible with int8 quantization and conversion. On longer sweeps, it avoided catastrophic collapse and stayed stable through 64, 128, and 256 tokens. Representative repaired metrics were:

- 64 tokens: global relative MSE about `0.283`, late/early ratio about `1.37`
- 128 tokens: global relative MSE about `0.304`, late/early ratio about `1.08`
- 256 tokens: global relative MSE about `0.313`, late/early ratio about `1.06`

This result matters because it showed the drift problem was not purely explosive. However, the path still had an error floor that was too high for a final quality claim.

### 6.2 Quantization was not the primary source of loss

A direct comparison between float and quantized versions of the repaired path showed that quantization error was much smaller than the overall approximation error. The quantized-vs-float delta at 256 tokens was around `0.011` relative MSE, implying that most remaining loss was architectural rather than a simple quantization artifact.

### 6.3 Signed-history corruption killed the manual buffer path

Basis-vector and signed-input tests showed that the Akida boundary treated the manually packed history asymmetrically:

- positive amplitudes mapped comparatively cleanly,
- small negative amplitudes collapsed into deadband-like behavior,
- larger negative amplitudes aliased into distorted nonzero codes.

An offset-push rescue improved the error by only about `1.9%`, far too little to justify continued investment in the manual history-buffer path. This branch was therefore killed for product use.

### 6.4 Akida-native temporal primitive preserves inhibitory sign

The decisive breakthrough came from a minimal signed-impulse probe using `DepthwiseBufferTempConv`. For input sequence `[-5, 0, 0, 0]`, the observed outputs were:

- float baseline: `[-5, -5, -5, 0]`
- quantized TensorFlow path: `[-4, -4, -4, 0]`
- Akida-converted path: `[-0.15625, -0.15625, -0.15625, 0.0]`

Although the amplitude was compressed, the crucial fact is that the **negative sign and temporal tail were preserved**. This cured the inhibition-blindness that invalidated the manual buffer path.

### 6.5 The high-level toolchain path is fragile but real

We found that the converter explicitly recognizes `QuantizedDepthwiseBufferTempConv > QuantizedReLU` as a compatible pattern. Several failed probes were eventually traced not to a dead architecture, but to a shape-contract mismatch: the temporal primitive expects streamed 4D inputs `(B, H, W, C)` and internally materializes FIFO state `(B, H, W, T, C)`. Feeding it an explicit time axis caused the rank-6 vs rank-5 failures seen in early attempts.

### 6.6 Kernel-4 is the currently validated fused operating point

Per-kernel probing found that `kernel_size = 4` is the only cleanly validated fused operating point in the current toolchain experiments. Kernels 2, 6, 8, and 10 failed later conversion or explicit quantized construction tests, even when the 4D streaming contract was corrected. Thus the present evidence supports `k = 4` as the current "silicon gold unit" for this toolchain build.

### 6.7 A stacked temporal bottleneck path converts cleanly

Most importantly, a conservative stacked temporal bottleneck block converted successfully:

- `16 -> 8 -> DepthwiseBufferTempConv(k=4) -> 16`
- repeated twice

The converted Akida model exposed **two fused `DepthwiseBufferTempConv` layers**, confirming that stacking is structurally viable. This establishes a path toward a temporal hierarchy rather than a single shallow temporal cell.

## 7. Discussion

### 7.1 What is proven

The present results support the following claim:

> A reliable structural conversion path exists for autoregressive temporal mixing on Akida-style spiking hardware, provided that the design uses hardware-native temporal FIFO primitives rather than manually packed signed history buffers.

This is a conversion-path claim, not yet a final LLM-quality claim.

### 7.2 What is not yet proven

We have **not** yet shown:

- end-to-end product-grade language quality,
- MSE below `0.15` on the final target task,
- long-horizon effective context comparable to a 96-token external buffer,
- or final spike-efficiency measurements on a production-equivalent hardware deployment.

### 7.3 Why the result still matters

In neuromorphic engineering, structural viability is often the hardest step. A design that cannot preserve inhibitory temporal information under conversion is not a candidate product path, regardless of float-domain behavior. By contrast, a design that preserves the right temporal sign structure and converts into stacked fused temporal cells is a strong base for further quality optimization.

## 8. Can this path plausibly reach relative MSE < 0.15?

Possibly, yes, but this remains a **research target**, not a demonstrated result.

The strongest evidence in favor is:

- the manual path’s dominant loss was architectural rather than quantization-limited,
- the native temporal primitive preserves inhibitory sign correctly,
- and stacked fused temporal blocks are now structurally viable.

The strongest caution is:

- the currently validated fused operating point is only `k = 4`,
- and the first stacked toy signed probe still showed behavioral fidelity issues despite structural conversion success.

Thus, dropping below `0.15` should be treated as plausible but unproven. The next engineering cycle must improve fidelity on the fused temporal stack, not merely prove more conversion variants.

## 9. Limitations

This draft has several important limitations.

1. The current fused temporal operating point is narrow (`k = 4`).
2. The tested stack is small and bottlenecked, not yet a realistic language-model block.
3. Current experiments use synthetic probes rather than a final language evaluation loop.
4. The current toolchain appears fragile across nominally supported FIFO sizes.
5. End-to-end energy claims require further hardware-side statistics and deployment validation.

## 10. Future Work

Immediate next steps:

1. Improve fidelity on the fused `k = 4` stacked path.
2. Test whether deeper stacking meaningfully extends useful temporal context.
3. Compare pointwise-temporal-pointwise bottlenecks against residualized short-window stacks.
4. Add spike/statistics reporting once the fused stack produces nontrivial signal under task-like inputs.
5. Evaluate cloud- or hardware-backed Akida 2.0 deployments once the fused stack passes the current fidelity gates.

## 11. Conclusion

The central result of this work is not that autoregressive LLMs already run well on spiking hardware. The result is that we have identified a **reliable structural conversion path** and separated it from misleading dead ends. Manual signed history buffers can appear numerically promising while failing at the hardware boundary. In contrast, Akida-native temporal FIFO primitives preserve inhibitory temporal behavior and can be stacked as fused hardware layers. This is the right substrate for future autoregressive spiking-language architectures.

## Appendix A. Reproducibility Notes

Key active files in `prototype-1` include:

- `experiments/kernel4_qdwbtc_min_probe.py`
- `experiments/kernel4_qdwbtc_double_stack_probe.py`
- `experiments/signed_impulse_depthwise_buffer_tempconv_min.py`
- `experiments/temporal_path_eval_contract.py`
- `artifacts/kernel4_qdwbtc_min_probe.json`
- `artifacts/kernel4_qdwbtc_double_stack_probe.json`
- `artifacts/signed_impulse_depthwise_buffer_tempconv_min.json`
- `artifacts/coordination_status.json`

Archived diagnostic material remains available under `prototype-1/archive/`.

## Suggested submission framing

Most defensible venues for the current state:

- AICAS
n- ISCAS workshop or special session
- Neuromorphic computing workshop tracks
- A technical note or systems-style short paper focused on conversion methodology

Least defensible framing right now:

- claiming full product-grade LLM quality
- claiming a final universal path for long-context autoregressive spiking LLMs
- claiming final hardware efficiency without completed hardware-side statistics
