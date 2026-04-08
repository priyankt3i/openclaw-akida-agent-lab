import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras
from tf_keras import Sequential, layers

SEED = 23
BATCH = 8
TOKENS = 128
DIM = 16
MODEL_HISTORY = 6
REPAIRED_HISTORY = 4
HIDDEN_1 = 64
HIDDEN_2 = 32
OUT_DIM = 16
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT = ARTIFACT_DIR / 'history_buffer_activation_audit.json'


def make_sequence(batch=BATCH, tokens=TOKENS, dim=DIM):
    rng = np.random.default_rng(SEED)
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


def keep_recent(flat_windows, model_history, keep_recent, dim):
    masked = flat_windows.reshape(flat_windows.shape[0], flat_windows.shape[1], model_history, dim).copy()
    if keep_recent < model_history:
        masked[:, :, : model_history - keep_recent, :] = 0.0
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


def per_token_activation_stats(model, flat_inputs, tokens, batch):
    relu_layers = [layer.output for layer in model.layers if isinstance(layer, layers.ReLU)]
    relu_defs = [layer for layer in model.layers if isinstance(layer, layers.ReLU)]
    probe = tf_keras.Model(model.input, relu_layers)
    outputs = probe(flat_inputs, training=False)
    if not isinstance(outputs, list):
        outputs = [outputs]

    report = {}
    for layer_def, layer_out in zip(relu_defs, outputs):
        arr = np.asarray(layer_out).reshape(batch, tokens, -1)
        report[layer_def.name] = {
            'zero_fraction_per_token': np.mean(arr == 0.0, axis=(0, 2)).tolist(),
            'sat_fraction_per_token': np.mean(arr >= 5.999, axis=(0, 2)).tolist(),
            'mean_per_token': np.mean(arr, axis=(0, 2)).tolist(),
            'var_per_token': np.var(arr, axis=(0, 2)).tolist(),
            'overall_zero_fraction': float(np.mean(arr == 0.0)),
            'overall_sat_fraction': float(np.mean(arr >= 5.999)),
            'overall_mean': float(np.mean(arr)),
            'overall_var': float(np.var(arr)),
        }
    return report


def summarize_trend(series):
    arr = np.asarray(series, dtype=np.float64)
    slope = float(np.polyfit(np.arange(arr.size), arr, deg=1)[0])
    early = float(np.mean(arr[: max(1, arr.size // 4)]))
    late = float(np.mean(arr[-max(1, arr.size // 4) :]))
    ratio = float(late / max(abs(early), 1e-12)) if early != 0 else float('inf')
    return {
        'slope': slope,
        'early_quarter_mean': early,
        'late_quarter_mean': late,
        'late_to_early_ratio': ratio,
    }


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    x = make_sequence()
    full_windows = causal_windows(x, MODEL_HISTORY)
    repaired_windows = keep_recent(full_windows, MODEL_HISTORY, REPAIRED_HISTORY, DIM)
    flat = repaired_windows.reshape(BATCH * TOKENS, MODEL_HISTORY * DIM)

    model = build_model()
    _ = model(flat, training=False).numpy()
    stats = per_token_activation_stats(model, flat, TOKENS, BATCH)

    report = {
        'seed': SEED,
        'batch': BATCH,
        'tokens': TOKENS,
        'dim': DIM,
        'model_history': MODEL_HISTORY,
        'repaired_history': REPAIRED_HISTORY,
        'model_name': model.name,
        'activation_layers': {},
    }

    for layer_name, layer_stats in stats.items():
        report['activation_layers'][layer_name] = {
            **layer_stats,
            'zero_fraction_trend': summarize_trend(layer_stats['zero_fraction_per_token']),
            'sat_fraction_trend': summarize_trend(layer_stats['sat_fraction_per_token']),
            'mean_trend': summarize_trend(layer_stats['mean_per_token']),
            'var_trend': summarize_trend(layer_stats['var_per_token']),
        }

    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
