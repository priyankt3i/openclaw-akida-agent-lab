# Prototype-1 One-Page Investor Summary

Date: 2026-04-08 UTC
Stage: Early technical validation, pre-hardware

## What we are building
A neuromorphic-friendly attention surrogate for transformer-like systems, designed to preserve useful attention behavior while making inference much more sparse, quantized, and hardware-deployable on BrainChip Akida-class tooling.

## Why it matters
Standard attention is dense and expensive. Neuromorphic hardware rewards sparse, event-driven computation, but standard softmax attention does not map cleanly to that world.

Our thesis is that a carefully designed sparse/spiking attention surrogate can preserve quality while materially reducing compute and energy.

## Best result so far
Current best software-surrogate operating point:
- Threshold: `0.20`
- Sparsity: `86.72%`
- Kept fraction: `13.28%`
- Efficiency gain proxy: `7.53x`
- Relative MSE: `0.266%`

Interpretation:
- This exceeds our internal target of `>=5x` efficiency improvement with `<1%` quality loss.
- This result is currently **software-surrogate only**, not yet hardware-validated.

## Progress versus baseline
- Initial baseline-like setup: about `1.13x` efficiency proxy
- Current best surrogate: `7.53x` efficiency proxy
- Improvement: about `6.67x` more efficiency gain proxy than the starting point

## What we have already de-risked
- Sparse threshold gating can beat the target in software
- Thresholding is a stronger neuromorphic fit than fixed top-k in our current setup
- Direct Akida conversion of standard transformer attention looks blocked
- An Akida-friendly surrogate path is more realistic than forcing exact softmax attention

## What is not yet proven
- Clean `cnn2snn` compatibility for the final surrogate
- Quantization and conversion of the Akida-friendly surrogate
- End-to-end Akida deployment
- Real measured joules on physical hardware

## Prior-art conclusion
The field already has meaningful prior work, including:
- Spikformer
- Spike-driven Transformer
- QKFormer
- Spikingformer
- SpikeBERT

So our value is **not** in claiming we invented spiking transformer attention.

Our value is more likely in:
- Akida-specific deployment realism
- explicit efficiency-vs-quality frontier tracking
- a practical path from surrogate math -> quantization -> conversion -> hardware

## Hardware recommendation today
Do **not** buy hardware yet.

We should consider buying the Akida board only after:
1. the Akida-friendly surrogate is implemented with supported primitives
2. compatibility checks pass
3. quantization and conversion succeed cleanly
4. the sparse win still survives after that

## Next milestone
Build the smallest `tf_keras` quantized gating surrogate using Akida-friendly primitives, then run:
- compatibility check
- quantization
- conversion

## Bottom line
This is already promising enough to keep funding and building.

But the honest claim today is:
- strong software evidence, yes
- differentiated deployment path potential, yes
- hardware proof, not yet
