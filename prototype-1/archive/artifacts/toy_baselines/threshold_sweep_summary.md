# Threshold sweep summary

- Command: `python prototype-1/experiments/threshold_sweep.py`
- Seed: `7`
- Quality target: relative MSE `< 0.01` (under 1% loss proxy)
- Gain target: `>= 5x`

## Best quality-constrained candidate

- Mode: `fixed_threshold`
- Threshold: `0.20`
- Kept fraction: `0.1328125`
- Sparsity: `0.8671875`
- Gain proxy: `7.529411764705882x`
- Relative MSE: `0.0026608324620881955` (~`0.266%`)
- Relative L2 error: `0.051583257575382624`
- Max absolute error: `1.9049373060068078`
- Mean kept per row: `1.0625`

## Highest gain overall

- Mode: `fixed_threshold`
- Threshold: `0.50`
- Gain proxy: `8.0x`
- Relative MSE: `0.020431983488056943` (~`2.043%`)
- Verdict: misses the <1% quality target

## Recommendation

Use threshold `0.20` as the current best tradeoff. It clears both targets, giving about `7.53x` efficiency-gain proxy at about `0.266%` relative MSE.
