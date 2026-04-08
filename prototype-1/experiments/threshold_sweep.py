import json
from pathlib import Path

import numpy as np

SEED = 7
BATCH = 2
TOKENS = 8
DIM = 16
OUT = Path("prototype-1/artifacts/threshold_sweep_results.json")

FIXED_THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]
ROW_MAX_RATIOS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
ROW_MEAN_OFFSETS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
QUALITY_TARGET_REL_MSE = 0.01
TARGET_GAIN = 5.0


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def dense_attention(x, wq, wk, wv):
    q = x @ wq
    k = x @ wk
    v = x @ wv
    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(q.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ v
    return out, weights


def apply_sparse_schedule(weights, mode, value):
    if mode == "fixed_threshold":
        threshold = float(value)
        mask = weights >= threshold
        schedule_info = {"threshold": threshold}
    elif mode == "row_max_ratio":
        ratio = float(value)
        threshold = np.max(weights, axis=-1, keepdims=True) * ratio
        mask = weights >= threshold
        schedule_info = {"row_max_ratio": ratio}
    elif mode == "row_mean_offset":
        offset = float(value)
        threshold = np.mean(weights, axis=-1, keepdims=True) + offset
        mask = weights >= threshold
        schedule_info = {"row_mean_offset": offset}
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    masked = np.where(mask, weights, 0.0)
    denom = np.sum(masked, axis=-1, keepdims=True)
    renorm = np.where(denom > 0, masked / denom, weights)
    fallback_rows = int(np.sum(denom <= 0))
    return renorm, mask, fallback_rows, schedule_info


def summarize_candidate(mode, value, dense_out, dense_weights):
    sparse_weights, mask, fallback_rows, schedule_info = apply_sparse_schedule(dense_weights, mode, value)
    sparse_out = sparse_weights @ dense_v

    mse = float(np.mean((sparse_out - dense_out) ** 2))
    rel_mse = float(mse / (np.mean(dense_out ** 2) + 1e-12))
    rel_l2 = float(np.linalg.norm(sparse_out - dense_out) / (np.linalg.norm(dense_out) + 1e-12))
    max_abs_err = float(np.max(np.abs(sparse_out - dense_out)))
    kept_fraction = float(np.mean(mask))
    sparsity = float(1.0 - kept_fraction)
    gain_proxy = float(1.0 / max(kept_fraction, 1e-12))

    row_kept = np.sum(mask, axis=-1)
    candidate = {
        "mode": mode,
        **schedule_info,
        "kept_fraction": kept_fraction,
        "sparsity": sparsity,
        "gain_proxy": gain_proxy,
        "mse": mse,
        "rel_mse": rel_mse,
        "rel_l2": rel_l2,
        "max_abs_err": max_abs_err,
        "fallback_rows": fallback_rows,
        "min_kept_per_row": int(np.min(row_kept)),
        "max_kept_per_row": int(np.max(row_kept)),
        "mean_kept_per_row": float(np.mean(row_kept)),
        "meets_quality_target": rel_mse <= QUALITY_TARGET_REL_MSE,
        "meets_gain_target": gain_proxy >= TARGET_GAIN,
    }
    candidate["meets_both_targets"] = bool(
        candidate["meets_quality_target"] and candidate["meets_gain_target"]
    )
    return candidate


rng = np.random.default_rng(SEED)
x = rng.standard_normal((BATCH, TOKENS, DIM), dtype=np.float32)
wq = rng.standard_normal((DIM, DIM), dtype=np.float32)
wk = rng.standard_normal((DIM, DIM), dtype=np.float32)
wv = rng.standard_normal((DIM, DIM), dtype=np.float32)

q = x @ wq
k = x @ wk
dense_v = x @ wv
scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(q.shape[-1])
dense_weights = softmax(scores, axis=-1)
dense_out = dense_weights @ dense_v

candidates = []
for threshold in FIXED_THRESHOLDS:
    candidates.append(summarize_candidate("fixed_threshold", threshold, dense_out, dense_weights))
for ratio in ROW_MAX_RATIOS:
    candidates.append(summarize_candidate("row_max_ratio", ratio, dense_out, dense_weights))
for offset in ROW_MEAN_OFFSETS:
    candidates.append(summarize_candidate("row_mean_offset", offset, dense_out, dense_weights))

quality_ok = [c for c in candidates if c["meets_quality_target"]]
frontier = sorted(
    quality_ok,
    key=lambda c: (-c["gain_proxy"], c["rel_mse"], c["fallback_rows"]),
)
summary = {
    "seed": SEED,
    "batch": BATCH,
    "tokens": TOKENS,
    "dim": DIM,
    "quality_target_rel_mse": QUALITY_TARGET_REL_MSE,
    "target_gain_proxy": TARGET_GAIN,
    "num_candidates": len(candidates),
    "best_quality_constrained": frontier[0] if frontier else None,
    "best_gain_overall": max(candidates, key=lambda c: c["gain_proxy"]),
    "candidates": candidates,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
