import csv
import json
from pathlib import Path

import numpy as np

SEED = 11
BATCH = 4
TOKENS = 32
DIM = 32
THRESHOLD = 0.20
OUT_DIR = Path("prototype-1/artifacts")


def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_mask(tokens):
    mask = np.triu(np.ones((tokens, tokens), dtype=bool), k=1)
    return mask


def dense_block(x, wq, wk, wv, wo):
    normed = layer_norm(x)
    q = normed @ wq
    k = normed @ wk
    v = normed @ wv
    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(q.shape[-1])
    mask = causal_mask(scores.shape[-1])
    scores = np.where(mask[None, :, :], -1e9, scores)
    weights = softmax(scores, axis=-1)
    attn_out = weights @ v
    proj = attn_out @ wo
    out = x + proj
    return out, weights, v


def apply_threshold(weights, threshold):
    mask = weights >= threshold
    identity = np.eye(weights.shape[-1], dtype=bool)[None, :, :]
    mask = np.logical_or(mask, identity)
    masked = np.where(mask, weights, 0.0)
    denom = np.sum(masked, axis=-1, keepdims=True)
    sparse = masked / np.maximum(denom, 1e-12)
    return sparse, mask


def summarize(dense_out, sparse_out, dense_weights, sparse_weights, mask):
    per_token_mse = np.mean((sparse_out - dense_out) ** 2, axis=(0, 2))
    per_token_rmse = np.sqrt(per_token_mse)
    per_token_rel_mse = np.mean((sparse_out - dense_out) ** 2, axis=(0, 2)) / (
        np.mean(dense_out ** 2, axis=(0, 2)) + 1e-12
    )
    attention_delta = np.mean(np.abs(sparse_weights - dense_weights), axis=0)
    row_keep_fraction = np.mean(mask, axis=(0, 2))
    prefix_rel_mse = []
    for end in range(1, dense_out.shape[1] + 1):
        dense_prefix = dense_out[:, :end, :]
        sparse_prefix = sparse_out[:, :end, :]
        mse = np.mean((sparse_prefix - dense_prefix) ** 2)
        rel = mse / (np.mean(dense_prefix ** 2) + 1e-12)
        prefix_rel_mse.append(float(rel))

    slope = float(np.polyfit(np.arange(dense_out.shape[1]), per_token_rel_mse, deg=1)[0])
    early = float(np.mean(per_token_rel_mse[: dense_out.shape[1] // 4]))
    late = float(np.mean(per_token_rel_mse[-dense_out.shape[1] // 4 :]))
    trend_ratio = float(late / max(early, 1e-12))

    return {
        "global_mse": float(np.mean((sparse_out - dense_out) ** 2)),
        "global_rel_mse": float(np.mean((sparse_out - dense_out) ** 2) / (np.mean(dense_out ** 2) + 1e-12)),
        "global_rel_l2": float(np.linalg.norm(sparse_out - dense_out) / (np.linalg.norm(dense_out) + 1e-12)),
        "kept_fraction": float(np.mean(mask)),
        "sparsity": float(1.0 - np.mean(mask)),
        "gain_proxy": float(1.0 / max(np.mean(mask), 1e-12)),
        "per_token_mse": per_token_mse.tolist(),
        "per_token_rmse": per_token_rmse.tolist(),
        "per_token_rel_mse": per_token_rel_mse.tolist(),
        "prefix_rel_mse": prefix_rel_mse,
        "row_keep_fraction": row_keep_fraction.tolist(),
        "attention_delta_heatmap": attention_delta.tolist(),
        "trend": {
            "per_token_rel_mse_slope": slope,
            "early_quarter_mean_rel_mse": early,
            "late_quarter_mean_rel_mse": late,
            "late_to_early_ratio": trend_ratio,
            "classification": classify_trend(slope, trend_ratio),
        },
    }


def classify_trend(slope, ratio):
    if ratio < 0.9 and slope < 0:
        return "decays_with_context"
    if ratio > 1.1 and slope > 0:
        return "grows_with_context"
    return "roughly_flat"


def write_csv(path, header, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((BATCH, TOKENS, DIM), dtype=np.float32)
    pos = np.linspace(-1.0, 1.0, TOKENS, dtype=np.float32)
    x = x + pos[None, :, None] * 0.15

    wq = rng.standard_normal((DIM, DIM), dtype=np.float32) / np.sqrt(DIM)
    wk = rng.standard_normal((DIM, DIM), dtype=np.float32) / np.sqrt(DIM)
    wv = rng.standard_normal((DIM, DIM), dtype=np.float32) / np.sqrt(DIM)
    wo = rng.standard_normal((DIM, DIM), dtype=np.float32) / np.sqrt(DIM)

    dense_out, dense_weights, v = dense_block(x, wq, wk, wv, wo)
    sparse_weights, mask = apply_threshold(dense_weights, THRESHOLD)
    sparse_out = x + (sparse_weights @ v) @ wo

    metrics = {
        "seed": SEED,
        "batch": BATCH,
        "tokens": TOKENS,
        "dim": DIM,
        "threshold": THRESHOLD,
    }
    metrics.update(summarize(dense_out, sparse_out, dense_weights, sparse_weights, mask))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "context_drift_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))

    write_csv(
        OUT_DIR / "context_drift_per_token.csv",
        ["token_idx", "rel_mse", "rmse", "row_keep_fraction"],
        [
            [idx, metrics["per_token_rel_mse"][idx], metrics["per_token_rmse"][idx], metrics["row_keep_fraction"][idx]]
            for idx in range(TOKENS)
        ],
    )

    heatmap_rows = []
    for q_idx, row in enumerate(metrics["attention_delta_heatmap"]):
        for k_idx, value in enumerate(row):
            heatmap_rows.append([q_idx, k_idx, value])
    write_csv(OUT_DIR / "context_drift_attention_delta_heatmap.csv", ["query_token", "key_token", "mean_abs_delta"], heatmap_rows)

    np.savez(
        OUT_DIR / "context_drift_arrays.npz",
        dense_out=dense_out,
        sparse_out=sparse_out,
        dense_weights=dense_weights,
        sparse_weights=sparse_weights,
        mask=mask.astype(np.int8),
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
