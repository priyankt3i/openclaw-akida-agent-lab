# Prototype-1 Founder Narrative Deck Outline

Date: 2026-04-08 UTC
Purpose: investor / partner deck skeleton grounded in current evidence

## Slide 1. Title
**Prototype-1: Toward Deployable Neuromorphic Attention**

Subtitle:
A sparse, quantization-aware attention surrogate for BrainChip Akida-class hardware

---

## Slide 2. The Problem
Transformers are powerful, but attention is expensive.

Key points:
- dense attention burns compute and energy
- neuromorphic hardware rewards sparse, event-driven computation
- standard softmax attention does not map cleanly onto that hardware

Message:
There is a gap between transformer capability and neuromorphic deployability.

---

## Slide 3. Our Thesis
We do not need to port full transformer attention literally.

We need to:
- preserve the useful behavior
- replace the expensive pieces
- make the computation sparse, quantized, and toolchain-compatible

Message:
We are targeting a deployable surrogate, not a cosmetic rewrite.

---

## Slide 4. Why Now
Why this is timely:
- energy efficiency is becoming a deployment bottleneck
- spiking and neuromorphic research is maturing
- but practical conversion to real hardware is still weak

Message:
The opportunity is not just better models. It is better deployment primitives.

---

## Slide 5. What We Built
Prototype-1 experiment stack includes:
- dense baseline attention harness
- sparse threshold-gating surrogate
- threshold sweep frontier
- top-k comparison
- spike/event proxy experiment
- Meta-TF / Akida feasibility pass
- Lava feasibility pass
- live mission dashboard and experiment log

Message:
This is already a real engineering loop, not just an idea.

---

## Slide 6. Best Result So Far
Headline metrics:
- Threshold `0.20`
- `86.72%` sparsity
- `13.28%` kept fraction
- `7.53x` efficiency gain proxy
- `0.266%` relative MSE

Message:
In software surrogate form, we already exceeded the internal target of >=5x efficiency gain with <1% quality loss.

Important footnote:
This is not yet a hardware-validated result.

---

## Slide 7. Why This Matters
Comparison story:
- baseline-like setup: `1.13x`
- best surrogate: `7.53x`
- top-k was more accurate but too dense
- event proxy was interesting but weaker

Message:
We are not just seeing random improvements, we are learning which mechanisms actually fit the neuromorphic objective.

---

## Slide 8. What We Learned
Key insights:
- thresholding beats fixed top-k on our current efficiency objective
- softmax is likely the wrong operation to force into Akida
- direct MultiHeadAttention conversion looks blocked
- the right path is an Akida-friendly surrogate built from supported primitives

Message:
The project has evolved from algorithm exploration into deployment-aware architecture design.

---

## Slide 9. Prior Art, Honestly
Prior art exists:
- Spikformer
- Spike-driven Transformer
- QKFormer
- Spikingformer
- SpikeBERT

Message:
We should not claim generic spiking-transformer novelty.

Better positioning:
- reproducible efficiency-quality frontier
- Akida-specific deployment path
- surrogate-to-hardware methodology

---

## Slide 10. Why We Still Matter
Differentiation thesis:
- most prior work focuses on model design and benchmark accuracy
- our value is deployment realism
- we are measuring whether the sparse gain survives actual toolchain constraints

Message:
This is closer to productizable neuromorphic inference than paper-only novelty.

---

## Slide 11. What Is De-risked vs Not
### De-risked
- sparse surrogate can beat internal target in software
- threshold gating looks promising
- the attention block can be made much sparser without immediate collapse

### Not de-risked
- Akida toolchain survival
- clean quantization/conversion
- real silicon energy numbers

Message:
We have crossed algorithmic plausibility, but not hardware proof.

---

## Slide 12. Hardware Purchase Policy
Current stance:
**Do not buy hardware yet.**

Buy only when:
1. surrogate uses supported primitives
2. compatibility passes
3. quantization and conversion succeed
4. gain still survives post-conversion

Message:
We are sequencing spend based on technical evidence, not excitement.

---

## Slide 13. Roadmap
Milestones:
1. Workspace and measurement harness, complete
2. Sparse surrogate discovery, complete
3. Alternative mechanism comparison, complete
4. Akida-friendly surrogate design, in progress
5. Quantization and conversion validation, pending
6. Hardware readiness decision, pending

Message:
We know exactly where we are and what remains.

---

## Slide 14. The Ask
Potential asks, depending on audience:
- time and runway to finish toolchain validation
- hardware budget after milestone 5 turns green
- strategic partnership around neuromorphic deployment
- technical advisory support on Akida conversion constraints

Message:
This is a disciplined technical wedge with clear next milestones.

---

## Slide 15. Closing
Final message:
We already have strong early evidence that a sparse neuromorphic attention surrogate can beat the efficiency target in software.

The question now is not whether the idea is interesting.
The question is whether we can make it survive real toolchain and hardware constraints.

That is exactly the kind of hard but valuable problem worth backing.
