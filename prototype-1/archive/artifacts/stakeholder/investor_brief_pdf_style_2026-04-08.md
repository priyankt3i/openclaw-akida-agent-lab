# Prototype-1
## Neuromorphic Attention Surrogate Brief

**Date:** 2026-04-08 UTC  
**Status:** Early technical validation, pre-hardware purchase  
**Project focus:** Sparse / spiking attention surrogate for BrainChip Akida-compatible deployment

---

## Executive Summary

Prototype-1 is an early-stage effort to redesign an attention-like transformer component so it better fits neuromorphic hardware constraints.

In current software-surrogate experiments, the best operating point has already achieved:

- **7.53x efficiency gain proxy**
- **0.266% relative MSE**
- **86.72% sparsity**

This is an encouraging result because it exceeds the internal target of at least **5x efficiency improvement** while remaining under the **1% quality-loss budget**.

However, this result is **not yet a hardware result**. The key remaining question is whether the same gain survives the real Akida toolchain, especially compatibility, quantization, and conversion.

---

## Problem

Transformer attention is powerful but expensive. Standard attention uses dense interactions and softmax normalization, both of which are a poor fit for neuromorphic execution.

BrainChip Akida and related toolchains work best with:
- quantized activations
- supported convolutional or dense-style primitives
- sparse, data-dependent activity
- minimal global coordination

This creates a practical gap between standard transformer research and deployable neuromorphic systems.

---

## Current Technical Result

### Best software-surrogate frontier point

- **Threshold:** `0.20`
- **Kept fraction:** `13.28%`
- **Sparsity:** `86.72%`
- **Efficiency gain proxy:** `7.53x`
- **Relative MSE:** `0.266%`
- **Relative L2:** `5.16%`

### Comparison to internal alternatives

| Experiment branch | Efficiency / Gain Proxy | Quality result | Interpretation |
|---|---:|---:|---|
| Baseline-like reference | `1.13x` | reference | Useful harness, not enough efficiency |
| Threshold gating, best | `7.53x` | `0.266%` relative MSE | Best current operating point |
| Threshold gating, lower threshold | `7.11x` | `0.054%` relative MSE | Better quality, slightly lower gain |
| Top-k (`k=2`) | `4.0x` | `3.32e-5` relative MSE | Very accurate, too dense |
| Spike/event proxy | `2.67x` | `0.809%` loss | Interesting, weaker than thresholding |
| Aggressive threshold (`0.50`) | `8.0x` | `2.04%` relative MSE | Better gain, misses quality target |

### Improvement over starting point

We improved from roughly **1.13x** efficiency proxy to **7.53x**, which is a large step-change in the internal surrogate benchmark.

---

## Prior Art and Positioning

We found credible prior work in spiking transformers and spike-native attention, including:
- **Spikformer**
- **Spike-driven Transformer**
- **QKFormer**
- **Spikingformer**
- **SpikeBERT**

### What this means

We should **not** position this as:
- “we invented spiking transformer attention”

We **can** position it as:
- an **Akida-oriented deployment path**
- a **reproducible efficiency-vs-quality frontier**
- a **toolchain-aware surrogate design methodology**
- a practical bridge from software surrogate to neuromorphic deployment

---

## What Has Been De-risked

### De-risked so far
- sparse threshold gating can exceed the internal efficiency target in software
- threshold gating appears more neuromorphic-friendly than fixed top-k in our current setup
- direct MultiHeadAttention conversion into the current Akida toolchain appears unrealistic
- an Akida-friendly surrogate built from supported primitives is the more plausible path

### Still at risk
- clean `cnn2snn` compatibility for the final surrogate
- quantization and conversion robustness
- hidden fallback paths or unsupported graph segments
- physical hardware validation

---

## Hardware Purchase Decision

### Recommendation today
**Do not purchase hardware yet.**

### Purchase trigger
We should revisit hardware purchase only after:
1. the Akida-friendly surrogate is implemented using supported primitives
2. compatibility checks pass
3. quantization and conversion succeed cleanly
4. the efficiency-quality win still survives after conversion

---

## Next Work Package

### Milestone 4 of 6, in progress
Design the Akida-friendly surrogate:
- replace softmax with an Akida-friendly gating function
- express projections with supported Dense or pointwise/separable-conv style primitives
- optionally test ternary or near-ternary gating if the conversion path stays clean

### Milestone 5 of 6, pending
Validate through the real toolchain:
- `cnn2snn.check_model_compatibility`
- quantization
- conversion
- post-conversion sanity checks

---

## Conclusion

Prototype-1 has already crossed an important threshold: the sparse neuromorphic-attention idea is no longer hypothetical.

The strongest current claim is:
- **the software surrogate is good enough to justify continued investment and deeper engineering**

The strongest honest limitation is:
- **the result is not yet proven on the Akida deployment path**

That makes this a compelling but still technical-risk-bearing opportunity.
