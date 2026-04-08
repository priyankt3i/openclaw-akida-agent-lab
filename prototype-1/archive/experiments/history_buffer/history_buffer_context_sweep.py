import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras
from tf_keras import Sequential, layers

SEED = 23
BATCH = 8
DIM = 16
MODEL_HISTORY = 6
REPAIRED_HISTORY = 4
CURRENT_ONLY_HISTORY = 1
HIDDEN_1 = 64
HIDDEN_2 = 32
OUT_DIM = 16
TOKEN_SWEEP = [64, 128, 256]
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT_JSON = ARTIFACT_DIR / 'history_buffer_context_sweep.json'
OUT_CSV = ARTIFACT_DIR / 'history_buffer_context_sweep.csv'


def make_sequence(tokens, batch=BATCH, dim=DIM):
    rng = np.random.default_rng(SEED + tokens)
    x = rng.integers(-16, 16, size=(batch, tokens, dim), dtype=np.int32).astype('float32')
    pos = np.linspace(-1.0, 1.0, tokens, dtype=np.float32)
    x = x + pos[None, :, None] * 1.5
    return x


def causal_windows(x, history):
    batch, tokens, dim = x.shape
    windows = np.zeros((batch, tokens, history, dim), dtype=np.float32)
    for t in range(tokens):
        start = max(0, t - history + 1)
        segment = x[:, start : t + 1, :]
        windows[:, t, -segment.shape[1] :, :] = segment
    return windows.reshape(batch, tokens, history * dim)


def keep_recent(flat_windows, model_history, keep_recent_tokens, dim):
    masked = flat_windows.reshape(flat_windows.shape[0], flat_windows.shape[1], model_history, dim).copy()
    if keep_recent_tokens < model_history:
        masked[:, :, : model_history - keep_recent_tokens, :] = 0.0
    return masked.reshape(flat_windows.shape)


def build_model():
    return Sequential([
        layers.Input(shape=(MODEL_HISTORY * DIM,), name='history_tokens_flat'),
        layers.Dense(HIDDEN_1, use_bias=False, name='proj_in'),
        layers.ReLU(max_value=6.0, name='gate_relu6'),
        layers.Dense(HIDDEN_2, use_bias=False, name='mix_dense'),
        layers.ReLU(max_value=6.0, name='post_relu6'),
        layers.Dense(OUT_DIM, use_bias=False, name='proj_out'),
    ], name='history_buffer_dense_relu6_surrogate')


def summarize(reference, candidate):
    diff = candidate - reference
    per_token_rel_mse = np.mean(diff ** 2, axis=(0, 2)) / (np.mean(reference ** 2, axis=(0, 2)) + 1e-12)
    prefix_rel_mse = []
    for end in range(1, reference.shape[1] + 1):
        rel = np.mean((candidate[:, :end, :] - reference[:, :end, :]) ** 2) / (np.mean(reference[:, :end, :] ** 2) + 1e-12)
        prefix_rel_mse.append(float(rel))
    x = np.arange(reference.shape[1], dtype=np.float64)
    slope = float(np.polyfit(x, per_token_rel_mse, deg=1)[0])
    early = float(np.mean(per_token_rel_mse[: reference.shape[1] // 4]))
    mid = float(np.mean(per_token_rel_mse[reference.shape[1] // 2 - reference.shape[1] // 8 : reference.shape[1] // 2 + reference.shape[1] // 8]))
    late = float(np.mean(per_token_rel_mse[-reference.shape[1] // 4 :]))
    late_to_early = float(late / max(early, 1e-12))
    mid_to_early = float(mid / max(early, 1e-12))
    first_half_slope = float(np.polyfit(x[: reference.shape[1] // 2], per_token_rel_mse[: reference.shape[1] // 2], deg=1)[0])
    second_half_slope = float(np.polyfit(x[reference.shape[1] // 2 :], per_token_rel_mse[reference.shape[1] // 2 :], deg=1)[0])
    slope_acceleration = float(second_half_slope / max(abs(first_half_slope), 1e-12))
    return {
        'global_rel_mse': float(np.mean(diff ** 2) / (np.mean(reference ** 2) + 1e-12)),
        'global_rel_l2': float(np.linalg.norm(diff) / (np.linalg.norm(reference) + 1e-12)),
        'per_token_rel_mse': per_token_rel_mse.tolist(),
        'prefix_rel_mse': prefix_rel_mse,
        'trend': {
            'per_token_rel_mse_slope': slope,
            'early_quarter_mean_rel_mse': early,
            'mid_band_mean_rel_mse': mid,
            'late_quarter_mean_rel_mse': late,
            'mid_to_early_ratio': mid_to_early,
            'late_to_early_ratio': late_to_early,
            'first_half_slope': first_half_slope,
            'second_half_slope': second_half_slope,
            'slope_acceleration_ratio': slope_acceleration,
        },
    }


def entropy_per_token(arr, bins=16):
    arr = np.asarray(arr)
    edges = np.linspace(0.0, 6.0, bins + 1)
    flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
    ent = []
    for t in range(flat.shape[1]):
        values = flat[:, t, :].reshape(-1)
        counts, _ = np.histogram(values, bins=edges)
        probs = counts.astype(np.float64) / max(np.sum(counts), 1)
        probs = probs[probs > 0]
        ent.append(float(-np.sum(probs * np.log2(probs))))
    return ent


def activation_stats(model, flat_inputs, tokens, batch):
    relu_layers = [layer.output for layer in model.layers if isinstance(layer, layers.ReLU)]
    relu_defs = [layer for layer in model.layers if isinstance(layer, layers.ReLU)]
    probe = tf_keras.Model(model.input, relu_layers)
    outputs = probe(flat_inputs, training=False)
    if not isinstance(outputs, list):
        outputs = [outputs]
    stats = {}
    for layer_def, layer_out in zip(relu_defs, outputs):
        arr = np.asarray(layer_out).reshape(batch, tokens, -1)
        var_per_token = np.var(arr, axis=(0, 2))
        ent_per_token = entropy_per_token(arr)
        early_var = float(np.mean(var_per_token[: tokens // 4]))
        late_var = float(np.mean(var_per_token[-tokens // 4 :]))
        early_ent = float(np.mean(ent_per_token[: tokens // 4]))
        late_ent = float(np.mean(ent_per_token[-tokens // 4 :]))
        stats[layer_def.name] = {
            'overall_zero_fraction': float(np.mean(arr == 0.0)),
            'overall_sat_fraction': float(np.mean(arr >= 5.999)),
            'var_trend': {
                'early_quarter_mean': early_var,
                'late_quarter_mean': late_var,
                'late_to_early_ratio': float(late_var / max(early_var, 1e-12)),
            },
            'entropy_trend': {
                'early_quarter_mean': early_ent,
                'late_quarter_mean': late_ent,
                'late_to_early_ratio': float(late_ent / max(early_ent, 1e-12)),
            },
        }
    return stats


def classify_state(repaired_metrics, activation_layer_stats):
    ratio = repaired_metrics['trend']['late_to_early_ratio']
    accel = repaired_metrics['trend']['slope_acceleration_ratio']
    post = activation_layer_stats['post_relu6']
    var_ratio = post['var_trend']['late_to_early_ratio']
    ent_ratio = post['entropy_trend']['late_to_early_ratio']
    if ratio > 4.0 or var_ratio < 0.85 or ent_ratio < 0.90:
        return 'suffocating'
    if ratio > 2.0 or accel > 1.5 or var_ratio < 0.95:
        return 'strained'
    return 'holding'


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()
    rows = []
    report = {
        'seed': SEED,
        'token_sweep': TOKEN_SWEEP,
        'model_name': model.name,
        'repaired_history': REPAIRED_HISTORY,
        'results': [],
    }

    for tokens in TOKEN_SWEEP:
        x = make_sequence(tokens)
        full_windows = causal_windows(x, MODEL_HISTORY)
        repaired_windows = keep_recent(full_windows, MODEL_HISTORY, REPAIRED_HISTORY, DIM)
        current_windows = keep_recent(full_windows, MODEL_HISTORY, CURRENT_ONLY_HISTORY, DIM)

        flat_full = full_windows.reshape(BATCH * tokens, MODEL_HISTORY * DIM)
        flat_repaired = repaired_windows.reshape(BATCH * tokens, MODEL_HISTORY * DIM)
        flat_current = current_windows.reshape(BATCH * tokens, MODEL_HISTORY * DIM)

        ref = model(flat_full, training=False).numpy().reshape(BATCH, tokens, OUT_DIM)
        repaired = model(flat_repaired, training=False).numpy().reshape(BATCH, tokens, OUT_DIM)
        current = model(flat_current, training=False).numpy().reshape(BATCH, tokens, OUT_DIM)

        repaired_metrics = summarize(ref, repaired)
        current_metrics = summarize(ref, current)
        activ_stats = activation_stats(model, flat_repaired, tokens, BATCH)
        state = classify_state(repaired_metrics, activ_stats)

        result = {
            'tokens': tokens,
            'current_only': current_metrics,
            'repaired_history_buffer': repaired_metrics,
            'activation_stats_repaired': activ_stats,
            'improvement': {
                'global_rel_mse_reduction_pct': float(
                    100.0 * (current_metrics['global_rel_mse'] - repaired_metrics['global_rel_mse'])
                    / max(current_metrics['global_rel_mse'], 1e-12)
                ),
                'late_quarter_rel_mse_reduction_pct': float(
                    100.0 * (
                        current_metrics['trend']['late_quarter_mean_rel_mse']
                        - repaired_metrics['trend']['late_quarter_mean_rel_mse']
                    ) / max(current_metrics['trend']['late_quarter_mean_rel_mse'], 1e-12)
                ),
            },
            'state_assessment': state,
        }
        report['results'].append(result)
        rows.append([
            tokens,
            current_metrics['global_rel_mse'],
            repaired_metrics['global_rel_mse'],
            repaired_metrics['trend']['late_to_early_ratio'],
            repaired_metrics['trend']['slope_acceleration_ratio'],
            activ_stats['post_relu6']['var_trend']['late_to_early_ratio'],
            activ_stats['post_relu6']['entropy_trend']['late_to_early_ratio'],
            state,
        ])

    OUT_JSON.write_text(json.dumps(report, indent=2))
    with OUT_CSV.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'tokens',
            'current_only_global_rel_mse',
            'repaired_global_rel_mse',
            'repaired_late_to_early_ratio',
            'repaired_slope_acceleration_ratio',
            'post_relu6_var_late_to_early_ratio',
            'post_relu6_entropy_late_to_early_ratio',
            'state_assessment',
        ])
        writer.writerows(rows)

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
