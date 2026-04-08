# Prior Art Summary for Prototype-1

Date: 2026-04-08 UTC
Purpose: concise summary of relevant papers, repositories, and what they mean for our work

## Executive summary

We found meaningful prior art in spiking transformers, spike-native self-attention, and early language-oriented spiking transformer variants.

So our work is **not novel** if framed as:
- "spiking transformer attention exists"
- "sparse/spiking attention can work"

However, our work can still be relevant and differentiated if framed as:
- **Akida-specific surrogate deployment path**
- **explicit efficiency-vs-quality tradeoff measurement**
- **conversion-aware design under real toolchain constraints**

## Papers and repositories found

### 1. Spikformer
- Paper: **Spikformer: When Spiking Neural Network Meets Transformer**
- Link: https://arxiv.org/abs/2209.15425
- Repo: https://github.com/ZK-Zhou/spikformer

What it does:
- Introduces Spiking Self Attention (SSA)
- Uses spike-form Q/K/V without softmax
- Focuses on vision and directly trained SNN transformer behavior

Why it matters for us:
- Confirms that softmax-free spiking attention is already an active area
- Reduces novelty if our pitch is just "spiking attention exists"
- Supports the idea that softmax replacement is the correct direction

### 2. Spike-driven Transformer
- Paper: **Spike-driven Transformer**
- Link: https://arxiv.org/abs/2307.01694

What it does:
- Pushes event-driven transformer computation
- Emphasizes binary spikes and data-dependent activity

Why it matters for us:
- Supports the importance of data-dependent thresholding over forced always-on mechanisms
- Makes our threshold-vs-top-k findings more believable in context

### 3. QKFormer
- Paper: **QKFormer: Hierarchical Spiking Transformer using Q-K Attention**
- Link: https://arxiv.org/abs/2403.16552
- Repo: https://github.com/zhouchenlin2096/QKFormer

What it does:
- Uses spike-form Q-K attention with linear-complexity binary vectors
- Focuses on high performance, especially on vision benchmarks like ImageNet

Why it matters for us:
- Shows the field is already moving away from exact softmax attention
- Supports our belief that a spike-native approximation is more realistic than literal transformer attention on neuromorphic hardware

### 4. Spikingformer
- Paper: **Spikingformer: A Key Foundation Model for Spiking Neural Networks**
- Link: https://arxiv.org/abs/2304.11954
- Repo: https://github.com/TheBrainLab/Spikingformer

What it does:
- Argues that earlier spiking transformer backbones still contain non-spike computation that hurts neuromorphic deployment
- Positions itself as a more deployment-realistic spiking backbone

Why it matters for us:
- This is especially relevant to our work
- It reinforces our current lesson that deployment realism matters more than abstract similarity to vanilla attention

### 5. SpikeBERT
- Repo: https://github.com/Lvchangze/SpikeBERT

What it does:
- Language-oriented spiking transformer variant learned from BERT with knowledge distillation

Why it matters for us:
- Confirms that language-oriented spiking transformer work already exists
- So our value cannot simply be "we applied spiking transformers to language"

## What the prior art means

### What is already crowded
- generic spiking transformer ideas
- spike-native attention approximations
- vision benchmark comparisons
- basic claims that spiking models can be more energy efficient

### What still looks open enough to matter
- BrainChip Akida-specific conversion-aware design
- small reproducible attention-block studies with explicit efficiency-quality frontiers
- methodology from dense reference -> sparse surrogate -> quantization -> conversion -> hardware

## Our current result versus internal alternatives

### Best current result
- Threshold `0.20`
- Sparsity `86.72%`
- Kept fraction `13.28%`
- Efficiency gain proxy `7.53x`
- Relative MSE `0.266%`

### Other internal references
- Lower-threshold threshold gating: `7.11x`, lower error
- Top-k (`k=2`): `4.0x`, much higher fidelity but misses current efficiency target
- Spike/event proxy: `2.67x`, under 1% loss but weaker than thresholding

## Honest competitiveness judgment

### Are others ahead on broad spiking transformer research?
Probably yes, especially on benchmarked architectures and accuracy-heavy vision results.

### Does that make our work irrelevant?
No.

### Why not?
Because our niche is narrower:
- can the sparse attention-like gain survive **Akida toolchain constraints**?
- can we build it from supported primitives?
- can we preserve a strong efficiency-quality tradeoff after quantization and conversion?

That is a more deployment-focused question than most of the public repos appear to answer.

## Best way to frame our work

### Bad framing
- "We invented spiking transformer attention"

### Better framing
- "We found a reproducible sparse attention surrogate that beats a 5x efficiency target in software while staying under a 1% quality-loss budget"

### Best framing, if we can finish the toolchain path
- "We built an Akida-friendly attention surrogate with a documented conversion path and verified efficiency-quality tradeoff"

## Recommendation

Keep going.

But keep the claims precise:
- strong software-surrogate result, yes
- full hardware-proof claim, not yet
- generic novelty claim, no
- deployment-specific differentiation potential, yes
