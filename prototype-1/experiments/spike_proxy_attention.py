import csv
import json
from pathlib import Path

import numpy as np

SEED = 11
BATCH = 2
TIMESTEPS = 8
TOKENS = 8
DIM = 16
DECAY = 0.75
DELTA_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
ROW_SPIKE_THRESHOLDS = [1, 2, 3, 4]
TOPK = [1, 2, 3]
ARTIFACT_DIR = Path('prototype-1/artifacts')
JSON_OUT = ARTIFACT_DIR / 'spike_proxy_metrics.json'
CSV_OUT = ARTIFACT_DIR / 'spike_proxy_sweep.csv'
TRACE_OUT = ARTIFACT_DIR / 'spike_proxy_traces.npz'


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def dense_attention_scores(x, wq, wk, wv):
    q = x @ wq
    k = x @ wk
    v = x @ wv
    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(q.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ v
    return scores, weights, v, out


def build_sequence(rng):
    base = rng.standard_normal((BATCH, TOKENS, DIM), dtype=np.float32)
    drift = rng.standard_normal((BATCH, TOKENS, DIM), dtype=np.float32) * 0.10
    jitter = rng.standard_normal((TIMESTEPS, BATCH, TOKENS, DIM), dtype=np.float32) * 0.035
    sequence = []
    for t in range(TIMESTEPS):
        phase = np.sin(np.float32((t + 1) / TIMESTEPS * np.pi))
        slow_component = phase * drift
        sequence.append(base + slow_component + jitter[t])
    return np.stack(sequence, axis=0)


def sparse_topk(weights, values, topk):
    ranks = np.argsort(weights, axis=-1)[..., ::-1]
    mask = np.zeros_like(weights, dtype=bool)
    for b in range(weights.shape[0]):
        for token in range(weights.shape[1]):
            selected = ranks[b, token, :topk]
            mask[b, token, selected] = True
    sparse = np.where(mask, weights, 0.0)
    denom = np.sum(sparse, axis=-1, keepdims=True)
    renorm = np.where(denom > 0, sparse / denom, weights)
    return renorm @ values, mask


def run_proxy(sequence, wq, wk, wv, delta_threshold, row_spike_threshold, topk):
    dense_outputs = []
    proxy_outputs = []
    dense_weights_all = []
    update_masks = []
    topk_masks = []
    dense_scores_all = []

    membrane = None
    prev_membrane = None
    prev_proxy = None
    value_compute = 0.0

    for t, x_t in enumerate(sequence):
        scores, weights, values, dense_out = dense_attention_scores(x_t, wq, wk, wv)
        sparse_out, topk_mask = sparse_topk(weights, values, topk)

        membrane = scores if membrane is None else DECAY * membrane + (1.0 - DECAY) * scores
        delta = np.zeros_like(membrane) if prev_membrane is None else membrane - prev_membrane
        spike_edges = np.abs(delta) >= delta_threshold
        row_spikes = np.sum(spike_edges, axis=-1)
        update_mask = row_spikes >= row_spike_threshold

        if prev_proxy is None:
            proxy_out = sparse_out
            update_mask = np.ones_like(update_mask, dtype=bool)
        else:
            proxy_out = np.where(update_mask[..., None], sparse_out, prev_proxy)

        dense_outputs.append(dense_out)
        proxy_outputs.append(proxy_out)
        dense_weights_all.append(weights)
        update_masks.append(update_mask)
        topk_masks.append(topk_mask)
        dense_scores_all.append(scores)

        value_compute += float(np.mean(update_mask) * (topk / TOKENS))
        prev_membrane = np.array(membrane, copy=True)
        prev_proxy = np.array(proxy_out, copy=True)

    dense_outputs = np.stack(dense_outputs)
    proxy_outputs = np.stack(proxy_outputs)
    dense_weights_all = np.stack(dense_weights_all)
    update_masks = np.stack(update_masks)
    topk_masks = np.stack(topk_masks)
    dense_scores_all = np.stack(dense_scores_all)

    mse = float(np.mean((proxy_outputs - dense_outputs) ** 2))
    rel_l2 = float(np.linalg.norm(proxy_outputs - dense_outputs) / (np.linalg.norm(dense_outputs) + 1e-12))
    quality_loss_pct = float(rel_l2 * 100.0)
    row_update_fraction = float(np.mean(update_masks))
    attention_kept_fraction = float(np.mean(topk_masks) * row_update_fraction)
    sparsity = float(1.0 - attention_kept_fraction)
    energy_gain = float(1.0 / max(attention_kept_fraction, 1e-12))
    attention_delta = float(np.mean(np.abs(proxy_outputs - dense_outputs)))

    return {
        'delta_threshold': delta_threshold,
        'row_spike_threshold': row_spike_threshold,
        'topk': topk,
        'mse': mse,
        'rel_l2': rel_l2,
        'quality_loss_pct': quality_loss_pct,
        'row_update_fraction': row_update_fraction,
        'attention_kept_fraction': attention_kept_fraction,
        'sparsity': sparsity,
        'energy_gain_proxy': energy_gain,
        'attention_mean_abs_delta': attention_delta,
        'value_compute_proxy': value_compute / TIMESTEPS,
        'dense_score_std': float(np.std(dense_scores_all)),
        'dense_weight_entropy': float(np.mean(-np.sum(dense_weights_all * np.log(dense_weights_all + 1e-12), axis=-1))),
        'trace': {
            'dense_outputs': dense_outputs,
            'proxy_outputs': proxy_outputs,
            'dense_weights': dense_weights_all,
            'update_masks': update_masks.astype(np.int8),
            'topk_masks': topk_masks.astype(np.int8),
            'dense_scores': dense_scores_all,
        },
    }


def main():
    rng = np.random.default_rng(SEED)
    sequence = build_sequence(rng)
    wq = rng.standard_normal((DIM, DIM), dtype=np.float32)
    wk = rng.standard_normal((DIM, DIM), dtype=np.float32)
    wv = rng.standard_normal((DIM, DIM), dtype=np.float32)

    sweep = []
    for delta_threshold in DELTA_THRESHOLDS:
        for row_spike_threshold in ROW_SPIKE_THRESHOLDS:
            for topk in TOPK:
                sweep.append(run_proxy(sequence, wq, wk, wv, delta_threshold, row_spike_threshold, topk))

    valid = [row for row in sweep if row['quality_loss_pct'] < 1.0]
    best = max(valid or sweep, key=lambda row: row['energy_gain_proxy'])

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'delta_threshold', 'row_spike_threshold', 'topk', 'mse', 'rel_l2', 'quality_loss_pct',
                'row_update_fraction', 'attention_kept_fraction', 'sparsity', 'energy_gain_proxy',
                'attention_mean_abs_delta', 'value_compute_proxy', 'dense_score_std', 'dense_weight_entropy'
            ],
        )
        writer.writeheader()
        for row in sweep:
            writer.writerow({k: v for k, v in row.items() if k != 'trace'})

    np.savez_compressed(TRACE_OUT, **best['trace'])

    metrics = {
        'seed': SEED,
        'batch': BATCH,
        'timesteps': TIMESTEPS,
        'tokens': TOKENS,
        'dim': DIM,
        'decay': DECAY,
        'searched_delta_thresholds': DELTA_THRESHOLDS,
        'searched_row_spike_thresholds': ROW_SPIKE_THRESHOLDS,
        'searched_topk': TOPK,
        'selected_delta_threshold': best['delta_threshold'],
        'selected_row_spike_threshold': best['row_spike_threshold'],
        'selected_topk': best['topk'],
        'mse': best['mse'],
        'rel_l2': best['rel_l2'],
        'quality_loss_pct': best['quality_loss_pct'],
        'row_update_fraction': best['row_update_fraction'],
        'attention_kept_fraction': best['attention_kept_fraction'],
        'sparsity': best['sparsity'],
        'energy_gain_proxy': best['energy_gain_proxy'],
        'attention_mean_abs_delta': best['attention_mean_abs_delta'],
        'value_compute_proxy': best['value_compute_proxy'],
        'dense_score_std': best['dense_score_std'],
        'dense_weight_entropy': best['dense_weight_entropy'],
        'sweep_rows': len(sweep),
        'quality_target_met': bool(best['quality_loss_pct'] < 1.0),
        'trace_file': str(TRACE_OUT),
        'sweep_file': str(CSV_OUT),
    }
    JSON_OUT.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
