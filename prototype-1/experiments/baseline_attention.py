import json
from pathlib import Path

import numpy as np

SEED = 7
BATCH = 2
TOKENS = 8
DIM = 16
THRESHOLD = 0.15
OUT = Path('prototype-1/artifacts/baseline_metrics.json')


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


def main():
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((BATCH, TOKENS, DIM), dtype=np.float32)
    wq = rng.standard_normal((DIM, DIM), dtype=np.float32)
    wk = rng.standard_normal((DIM, DIM), dtype=np.float32)
    wv = rng.standard_normal((DIM, DIM), dtype=np.float32)

    out, weights = dense_attention(x, wq, wk, wv)

    sparsity = float(np.mean(weights < THRESHOLD))
    max_weight = float(np.max(weights))
    mean_weight = float(np.mean(weights))
    entropy = -np.sum(weights * np.log(weights + 1e-12), axis=-1)
    mean_entropy = float(np.mean(entropy))

    metrics = {
        "seed": SEED,
        "batch": BATCH,
        "tokens": TOKENS,
        "dim": DIM,
        "threshold": THRESHOLD,
        "sparsity": sparsity,
        "max_weight": max_weight,
        "mean_weight": mean_weight,
        "mean_entropy": mean_entropy,
        "out_mean": float(np.mean(out)),
        "out_std": float(np.std(out)),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
