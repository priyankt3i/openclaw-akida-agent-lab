import json
from pathlib import Path

import numpy as np

SEED = 7
BATCH = 2
TOKENS = 8
DIM = 16
THRESHOLD = 0.15
OUT = Path('prototype-1/artifacts/sparse_metrics.json')


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


def sparse_attention(x, wq, wk, wv, threshold):
    q = x @ wq
    k = x @ wk
    v = x @ wv
    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(q.shape[-1])
    weights = softmax(scores, axis=-1)
    mask = weights >= threshold
    # Renormalize masked weights to keep stochasticity.
    masked = np.where(mask, weights, 0.0)
    denom = np.sum(masked, axis=-1, keepdims=True)
    # Avoid divide-by-zero: if a row is fully masked, fall back to dense weights.
    renorm = np.where(denom > 0, masked / denom, weights)
    out = renorm @ v
    return out, weights, renorm, mask


def main():
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((BATCH, TOKENS, DIM), dtype=np.float32)
    wq = rng.standard_normal((DIM, DIM), dtype=np.float32)
    wk = rng.standard_normal((DIM, DIM), dtype=np.float32)
    wv = rng.standard_normal((DIM, DIM), dtype=np.float32)

    dense_out, dense_weights = dense_attention(x, wq, wk, wv)
    sparse_out, dense_w, sparse_w, mask = sparse_attention(x, wq, wk, wv, THRESHOLD)

    # Quality metrics vs baseline.
    mse = float(np.mean((sparse_out - dense_out) ** 2))
    denom = float(np.mean(dense_out ** 2) + 1e-12)
    rel_mse = float(mse / denom)
    max_abs_err = float(np.max(np.abs(sparse_out - dense_out)))

    # Sparsity and energy proxy.
    sparsity = float(np.mean(dense_w < THRESHOLD))
    kept_fraction = float(np.mean(mask))
    energy_proxy = kept_fraction  # proportional to active weights

    metrics = {
        "seed": SEED,
        "batch": BATCH,
        "tokens": TOKENS,
        "dim": DIM,
        "threshold": THRESHOLD,
        "baseline_sparsity": sparsity,
        "kept_fraction": kept_fraction,
        "energy_proxy": energy_proxy,
        "mse": mse,
        "rel_mse": rel_mse,
        "max_abs_err": max_abs_err,
        "dense_out_mean": float(np.mean(dense_out)),
        "dense_out_std": float(np.std(dense_out)),
        "sparse_out_mean": float(np.mean(sparse_out)),
        "sparse_out_std": float(np.std(sparse_out)),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
