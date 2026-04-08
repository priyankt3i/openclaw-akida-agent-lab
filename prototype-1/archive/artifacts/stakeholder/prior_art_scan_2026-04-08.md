# Prior Art Scan for Prototype-1

Date: 2026-04-08 UTC

## Scope

This scan focused on:
- spiking transformer / spiking self-attention papers
- public code repositories for spiking transformer models
- Akida relevance
- Lava relevance
- whether our current sparse-threshold surrogate result looks ordinary, competitive, or differentiated

## Key external references found

### 1. Spikformer (2022/2023)
- Paper: "Spikformer: When Spiking Neural Network Meets Transformer"
- arXiv: https://arxiv.org/abs/2209.15425
- Public repo: https://github.com/ZK-Zhou/spikformer

Main signal:
- Introduces Spiking Self Attention (SSA) using spike-form Q/K/V without softmax.
- Focus is end-to-end spiking transformer modeling, mainly vision benchmarks.
- Reported as energy-efficient and multiplication-light, but framed around model design and accuracy rather than BrainChip Akida deployment.

### 2. Spike-driven Transformer (2023)
- Paper: "Spike-driven Transformer"
- arXiv: https://arxiv.org/abs/2307.01694 (title matched in arXiv search as spike-driven transformer family)

Main signal:
- Explicitly pushes event-driven transformer behavior.
- Claims event-driven sparsity and binary spike computation.
- Again, the emphasis is architectural design and general neuromorphic efficiency, not Akida conversion.

### 3. QKFormer (2024)
- Paper: "QKFormer: Hierarchical Spiking Transformer using Q-K Attention"
- arXiv: https://arxiv.org/abs/2403.16552
- Public repo: https://github.com/zhouchenlin2096/QKFormer

Main signal:
- Replaces full attention with spike-form Q-K attention using linear-complexity binary vectors.
- Strong performance focus, especially on ImageNet-1k.
- Important because it shows the field is already moving away from exact softmax toward more spike-native attention approximations.

### 4. Spikingformer (AAAI 2026)
- Paper: "Spikingformer: A Key Foundation Model for Spiking Neural Networks"
- arXiv: https://arxiv.org/abs/2304.11954
- Public repo: https://github.com/TheBrainLab/Spikingformer

Main signal:
- Explicitly argues that earlier spiking transformer backbones still contain non-spike computations that hurt neuromorphic deployment.
- This is highly relevant to us because it reinforces the same lesson we found: deployment realism matters more than abstract architectural similarity to vanilla transformers.

### 5. SpikeBERT
- Public repo: https://github.com/Lvchangze/SpikeBERT

Main signal:
- Shows there is already at least some public work on language-oriented spiking transformer variants learned from BERT.
- That reduces novelty if our only claim is "spiking transformer for language exists."
- It does not invalidate our work, but it means our value has to come from hardware/toolchain realism, efficiency tradeoff, or deployment path.

## What the field seems to be doing already

The prior art suggests the community already has:
- multiple spiking-transformer architectures
- spike-native alternatives to softmax attention
- strong vision-oriented benchmarks
- early language-oriented spiking transformer work

So our result is **not** novel if framed only as:
- "we made a sparse/spiking transformer attention-like mechanism"

That alone would be crowded.

## Where our work can still be relevant

Our result still looks relevant if we frame it as one of these:

### A. Akida-oriented deployment path
This is the strongest differentiator.
Most public spiking-transformer work appears focused on:
- direct training
- PyTorch/SpikingJelly style research stacks
- accuracy on standard datasets

What we are doing instead is asking:
- can a sparse attention-like surrogate survive BrainChip Akida toolchain constraints?
- can it be quantized and converted with supported primitives?
- can we preserve a concrete quality-efficiency win under those constraints?

That is much more deployment-specific.

### B. Reproducible efficiency-vs-quality frontier on a small attention block
Our current best result is:
- threshold `0.20`
- sparsity `0.8671875`
- kept fraction `0.1328125`
- efficiency gain proxy `7.53x`
- relative MSE `0.266%`

This is interesting if presented as:
- a reproducible operating-point search on an attention-like block
- with explicit quality and efficiency tradeoff tracking
- targeted toward neuromorphic deployment constraints

### C. Surrogate-to-hardware translation methodology
A lot of papers jump from architectural claims to benchmark results.
If we can show a practical methodology like:
1. dense reference
2. sparse/spike surrogate frontier
3. hardware-compatible primitive substitution
4. quantization/conversion
5. final deployment

then that process itself can be valuable even if the raw idea overlaps with existing literature.

## My judgment

### Is our current result irrelevant because others have done better?
No.

### Is it automatically novel?
Also no.

### Current honest assessment
- As a pure "spiking transformer exists" claim, we are not novel.
- As a pure accuracy benchmark claim, we are probably not yet competitive with the stronger spiking-transformer papers.
- As an Akida-specific sparse-attention deployment path with explicit efficiency/quality tradeoff tracking, we may still have a meaningful niche.

## Most important conclusion

Our current result is **interesting enough to keep going**, but only if we position it correctly.

Bad framing:
- "we invented spiking transformer attention"

Better framing:
- "we found a reproducible sparse attention surrogate that exceeds a 5x efficiency target in software, and we are testing whether that win survives Akida-compatible conversion"

Best framing, if we can deliver it:
- "we built an Akida-friendly attention surrogate with a verified efficiency-quality tradeoff and a documented conversion path"

## What would make this clearly stronger

1. Prove the Akida-compatible surrogate path works.
2. Show the efficiency-quality frontier on a slightly larger, less quantized setup.
3. Compare against at least one spike-native alternative such as top-k or Q-K style linear attention.
4. Demonstrate why our path is more deployment-realistic than direct softmax-style spiking attention.

## Recommendation

Keep going.

But do **not** treat the current software surrogate win as enough by itself.
The differentiator must come from deployment realism, toolchain compatibility, and reproducible efficiency-quality tradeoff evidence.
