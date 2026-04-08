# Investor Feedback Action Plan

Date: 2026-04-08 UTC

## What the critique got right

1. **"Proxy gain" is not enough**
   - Correct. Our `7.53x` figure is currently a software-surrogate gain proxy, not measured joules on hardware.
   - We should stop treating that as sufficient proof.

2. **Softmax being a bad Akida fit is not a novel insight**
   - Correct. That is a known constraint, not our discovery.
   - Our differentiation must come from the specific replacement path that survives quantization and conversion.

3. **Deployment realism only matters if the surrogate survives the real toolchain**
   - Correct. If the surrogate collapses into a generic conv-like block with no distinctive global-context behavior, the value weakens substantially.

4. **TENN is a serious competitive threat**
   - Correct. If BrainChip already has a native sequence alternative, our niche must be sharper.

## What our moat has to become

Not:
- "we made a sparse transformer"

Instead:
- "we have an attention-like surrogate that survives Akida quantization and conversion while preserving useful global-context behavior better than simpler alternatives"

## Immediate strategic changes

### 1. Replace "proxy gain" with stronger evidence
Target outputs to add next:
- quantized error tables at 8-bit / 4-bit / 2-bit where possible
- conversion survival status
- toolchain-compatible graph proof
- if available later, cycle-accurate or hardware-estimated joules/inference

### 2. Focus the moat on the quantization bridge
Questions that matter now:
- Does the surrogate still work at low bit width?
- Does it survive `cnn2snn` cleanly?
- Does it preserve quality better than other sparse alternatives after quantization?

### 3. Answer the TENN challenge directly
We need a clearer comparison around:
- global context retention
- long-range interaction behavior
- whether our surrogate preserves attention-like selectivity in ways a state-space or polynomial-decay alternative may not

### 4. Produce more diagnostic evidence
Needed analyses:
- row-wise / token-wise MSE breakdown on the current surrogate
- error heatmap for dense vs surrogate attention outputs
- a comparison against a more GPT-2-like single attention block
- if possible, per-stage error after quantization

## Practical next experiments

1. Build the minimal Akida-friendly `tf_keras` gating surrogate.
2. Replace softmax with quantization-friendly gating.
3. Run `cnn2snn.check_model_compatibility`.
4. Attempt quantization and conversion.
5. Produce error breakdowns, not just aggregate relative MSE.
6. Validate the surrogate on a slightly larger, less toy attention block.
7. Compare against a simple GPT-2-style attention reference block.

## Current honest answer to the investor question

Can we show a confusion matrix or MSE-per-layer for a GPT-2 style attention block right now?
- **Not yet in the form they want.**
- We currently have aggregate error on a small deterministic attention-like surrogate, not a full GPT-2 layer study.
- We can produce a more detailed error analysis next, and we should.

## Conclusion

The critique is useful.
It does not kill the project, but it forces a better claim:

- software surrogate success is real
- hardware moat is not proven yet
- quantization-and-conversion survival is now the central proof obligation
