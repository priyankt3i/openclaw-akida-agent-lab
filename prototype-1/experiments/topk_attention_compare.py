import json
from pathlib import Path

import numpy as np

SEED = 7
BATCH = 2
TOKENS = 8
DIM = 16
THRESHOLD = 0.15
TOP_K = 2
OUT = Path("prototype-1/artifacts/topk_compare_metrics.json")


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
    return out, weights, v


def apply_threshold(weights, threshold):
    mask = weights >= threshold
    masked = np.where(mask, weights, 0.0)
    denom = np.sum(masked, axis=-1, keepdims=True)
    renorm = np.where(denom > 0, masked / denom, weights)
    return renorm, mask


def apply_topk(weights, top_k):
    topk_idx = np.argpartition(weights, -top_k, axis=-1)[..., -top_k:]
    mask = np.zeros_like(weights, dtype=bool)
    np.put_along_axis(mask, topk_idx, True, axis=-1)
    masked = np.where(mask, weights, 0.0)
    denom = np.sum(masked, axis=-1, keepdims=True)
    renorm = masked / np.maximum(denom, 1e-12)
    return renorm, mask


def summarize(name, sparse_out, dense_out, dense_weights, sparse_weights, mask):
    mse = float(np.mean((sparse_out - dense_out) ** 2))
    rel_mse = float(mse / (np.mean(dense_out ** 2) + 1e-12))
    rel_l2 = float(np.linalg.norm(sparse_out - dense_out) / (np.linalg.norm(dense_out) + 1e-12))
    attention_mae = float(np.mean(np.abs(sparse_weights - dense_weights)))
    kept_fraction = float(np.mean(mask))
    sparsity = float(1.0 - kept_fraction)
    energy_proxy = kept_fraction
    efficiency_gain = float(1.0 / max(energy_proxy, 1e-12))
    return {
        "name": name,
        "kept_fraction": kept_fraction,
        "sparsity": sparsity,
        "energy_proxy": energy_proxy,
        "efficiency_gain_proxy": efficiency_gain,
        "mse": mse,
        "rel_mse": rel_mse,
        "rel_l2": rel_l2,
        "attention_mae": attention_mae,
        "max_abs_err": float(np.max(np.abs(sparse_out - dense_out))),
        "out_mean": float(np.mean(sparse_out)),
        "out_std": float(np.std(sparse_out)),
    }


def main():
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((BATCH, TOKENS, DIM), dtype=np.float32)
    wq = rng.standard_normal((DIM, DIM), dtype=np.float32)
    wk = rng.standard_normal((DIM, DIM), dtype=np.float32)
    wv = rng.standard_normal((DIM, DIM), dtype=np.float32)

    dense_out, dense_weights, v = dense_attention(x, wq, wk, wv)

    threshold_weights, threshold_mask = apply_threshold(dense_weights, THRESHOLD)
    threshold_out = threshold_weights @ v

    topk_weights, topk_mask = apply_topk(dense_weights, TOP_K)
    topk_out = topk_weights @ v

    metrics = {
        "seed": SEED,
        "batch": BATCH,
        "tokens": TOKENS,
        "dim": DIM,
        "threshold": THRESHOLD,
        "top_k": TOP_K,
        "dense": {
            "out_mean": float(np.mean(dense_out)),
            "out_std": float(np.std(dense_out)),
            "attention_entropy": float(np.mean(-np.sum(dense_weights * np.log(dense_weights + 1e-12), axis=-1))),
        },
        "threshold": summarize("threshold", threshold_out, dense_out, dense_weights, threshold_weights, threshold_mask),
        "topk": summarize("topk", topk_out, dense_out, dense_weights, topk_weights, topk_mask),
    }
    metrics["tradeoff"] = {
        "topk_vs_threshold_rel_mse_ratio": float(metrics["topk"]["rel_mse"] / max(metrics["threshold"]["rel_mse"], 1e-12)),
        "topk_vs_threshold_efficiency_ratio": float(metrics["topk"]["efficiency_gain_proxy"] / max(metrics["threshold"]["efficiency_gain_proxy"], 1e-12)),
        "recommended": "topk" if metrics["topk"]["rel_mse"] <= metrics["threshold"]["rel_mse"] and metrics["topk"]["efficiency_gain_proxy"] >= metrics["threshold"]["efficiency_gain_proxy"] else "threshold",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
