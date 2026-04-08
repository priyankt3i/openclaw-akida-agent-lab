# Prototype-1 Investor Pitch Brief

Date: 2026-04-08 UTC
Status: Early technical validation, not yet hardware-validated

## One-line summary

We have built an early neuromorphic-attention prototype that, in software surrogate experiments, achieved a **7.53x efficiency gain proxy** while keeping **relative MSE to about 0.266%**, beating our internal target of **>=5x efficiency improvement with <1% quality loss**.

## The problem

Modern transformer attention is dense and energy-hungry. Neuromorphic hardware, especially BrainChip Akida-class deployment targets, rewards sparse, event-driven, quantized computation. The challenge is that standard attention relies on operations like **softmax** and global token interaction, which are poorly matched to neuromorphic toolchains.

## What we are building

We are developing a **spiking / sparse attention surrogate** that preserves the useful behavior of transformer attention while being more compatible with neuromorphic deployment constraints.

The current focus is **not** a full LLM replacement. It is a targeted attack on one expensive component, attention-like mixing, with the goal of proving a measurable energy-efficiency advantage before hardware purchase.

## Best result so far

### Current best software surrogate operating point
- Threshold: `0.20`
- Sparsity: `86.72%`
- Kept fraction: `13.28%`
- Efficiency gain proxy: `7.53x`
- Relative MSE: `0.266%`
- Relative L2 error: `5.16%`

Interpretation:
- This exceeds the project target on the current small deterministic surrogate.
- It suggests the sparse-attention idea is not just plausible, it is already measurable.
- The remaining technical risk is translation into the Akida toolchain, not whether sparse behavior helps at all.

## Comparison versus other internal experiment branches

| Method | Efficiency / Gain Proxy | Quality Metric | Takeaway |
|---|---:|---:|---|
| Dense baseline reference | `1.13x` | baseline reference | Harness works, but no meaningful efficiency win |
| Threshold-gated sparse surrogate, best | `7.53x` | `0.266%` relative MSE | Best current operating point |
| Threshold-gated sparse surrogate, lower threshold | `7.11x` | `0.054%` relative MSE | Better quality, slightly lower gain |
| Top-k gating (`k=2`) | `4.0x` | `3.32e-5` relative MSE | Excellent fidelity, too dense for current target |
| Spike/event proxy | `2.67x` | `0.809%` quality loss | Interesting but weaker than simple thresholding |
| Aggressive threshold (`0.50`) | `8.0x` | `2.04%` relative MSE | Better gain, misses quality target |

## Improvement over where we started

We improved from an initial baseline-like setup with only about **1.13x** efficiency proxy to a reproducible threshold-gated operating point at **7.53x**.

That is roughly:
- **6.67x more efficiency gain proxy** than the initial baseline-like setup, or
- an increase of about **567%** over the initial proxy value, depending on how you present it.

More importantly, the best result stayed under the project quality-loss budget.

## Why this is still interesting even though prior art exists

We found existing prior art in spiking transformers and spiking self-attention, including:
- **Spikformer**
- **Spike-driven Transformer**
- **QKFormer**
- **Spikingformer**
- **SpikeBERT**

So this is **not** a claim that we invented spiking transformer attention.

The differentiator we are pursuing is narrower and more practical:
- explicit **Akida-oriented deployment path**
- reproducible **efficiency-vs-quality frontier tracking**
- conversion-aware surrogate design using supported primitives
- a path from software surrogate -> quantization -> conversion -> hardware

That is a stronger commercial story than a generic "we have a spiking transformer" claim.

## What has been de-risked

### De-risked
- Sparse attention-like behavior can exceed the internal 5x target in software.
- Thresholding is a better neuromorphic fit than fixed top-k in our current setup.
- Direct transformer attention conversion is probably the wrong path.
- An Akida-friendly surrogate path appears more realistic than literal softmax attention.

### Not yet de-risked
- Successful `cnn2snn` compatibility for the final surrogate
- Quantization and conversion of the new surrogate path
- End-to-end Akida deployment
- Real measured joules on physical silicon

## Hardware purchase status

### Recommendation today
**Do not buy hardware yet.**

Why:
- the software result is promising
- but the toolchain-valid surrogate still needs to survive compatibility, quantization, and conversion
- direct MultiHeadAttention conversion looks blocked

### Purchase trigger
We should consider buying hardware only after:
1. the Akida-friendly surrogate is implemented with supported primitives
2. it passes compatibility checks
3. quantization and conversion succeed cleanly
4. the sparse win still looks strong after that

## What comes next

### Milestone 4 of 6, in progress
Akida-friendly surrogate design:
- replace softmax with an Akida-friendly gate such as ReLU6 clamp or similar quantized gating
- express projections with supported Dense or pointwise/separable-conv style primitives
- optionally test ternary or near-ternary gating if the conversion path stays clean

### Milestone 5 of 6, pending
Quantization and conversion validation:
- run `cnn2snn.check_model_compatibility`
- quantize
- convert
- verify no fatal fallback or incompatibility

## The investor-relevant thesis

If this works through the real toolchain, the opportunity is not just a benchmark curiosity.
It is a **deployment-oriented neuromorphic inference primitive** for transformer-like systems, with a methodology that is:
- sparse by design
- quantization-aware
- hardware-conscious
- measurable in terms of efficiency vs quality

## Current bottom line

- **Technical direction:** promising
- **Best software result:** strong
- **Novelty claim:** not generic novelty, but potentially differentiated through Akida deployment realism
- **Hardware readiness:** not yet
- **Why continue:** because we already crossed the target in the surrogate, and the remaining question is now implementation realism rather than algorithmic plausibility
