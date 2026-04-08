import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from cnn2snn import quantize
from tf_keras import Sequential, layers

SEED = 23
TOKENS = 32
DIM = 16
MODEL_HISTORY = 6
HIDDEN_1 = 64
HIDDEN_2 = 32
OUT_DIM = 16
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT = ARTIFACT_DIR / 'history_buffer_quant_audit.json'


def build_model():
    return Sequential([
        layers.Input(shape=(MODEL_HISTORY * DIM,), name='history_tokens_flat'),
        layers.Dense(HIDDEN_1, use_bias=False, name='proj_in'),
        layers.ReLU(max_value=6.0, name='gate_relu6'),
        layers.Dense(HIDDEN_2, use_bias=False, name='mix_dense'),
        layers.ReLU(max_value=6.0, name='post_relu6'),
        layers.Dense(OUT_DIM, use_bias=False, name='proj_out'),
    ], name='history_buffer_dense_relu6_surrogate')


def make_sample(batch=8):
    rng = np.random.default_rng(SEED)
    x = rng.integers(-16, 16, size=(batch, TOKENS, DIM), dtype=np.int32).astype('float32')
    pos = np.linspace(-1.0, 1.0, TOKENS, dtype=np.float32)
    x = x + pos[None, :, None] * 1.5
    windows = np.zeros((batch, TOKENS, MODEL_HISTORY, DIM), dtype=np.float32)
    for t in range(TOKENS):
        start = max(0, t - MODEL_HISTORY + 1)
        segment = x[:, start : t + 1, :]
        windows[:, t, -segment.shape[1] :, :] = segment
    return windows.reshape(batch * TOKENS, MODEL_HISTORY * DIM)


def summarize_tensor(arr):
    flat = np.asarray(arr).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {'size': int(flat.size), 'finite': 0}

    rounded = np.round(finite)
    is_integer_like = bool(np.allclose(finite, rounded, atol=1e-6))
    if is_integer_like:
        vals = rounded.astype(np.int32)
        neg_rail_frac = float(np.mean(vals <= -127))
        pos_rail_frac = float(np.mean(vals >= 127))
        edge_mass_frac = float(np.mean((vals <= -120) | (vals >= 120)))
        unique, counts = np.unique(vals, return_counts=True)
        top_idx = np.argsort(counts)[-10:][::-1]
        top_values = [
            {'value': int(unique[i]), 'count': int(counts[i]), 'fraction': float(counts[i] / vals.size)}
            for i in top_idx
        ]
    else:
        vals = finite.astype(np.float32)
        neg_rail_frac = 0.0
        pos_rail_frac = 0.0
        edge_mass_frac = 0.0
        hist_counts, hist_edges = np.histogram(vals, bins=16)
        top_values = {
            'hist_edges': hist_edges.tolist(),
            'hist_counts': hist_counts.tolist(),
        }

    return {
        'dtype': str(np.asarray(arr).dtype),
        'size': int(flat.size),
        'is_integer_like': is_integer_like,
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'mean': float(np.mean(finite)),
        'std': float(np.std(finite)),
        'neg_rail_fraction': neg_rail_frac,
        'pos_rail_fraction': pos_rail_frac,
        'edge_mass_fraction_abs_ge_120': edge_mass_frac,
        'top_values': top_values,
    }


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()
    sample = make_sample()
    _ = model(sample, training=False).numpy()
    qmodel = quantize(model, input_weight_quantization=8, weight_quantization=4, activ_quantization=4)

    report = {
        'seed': SEED,
        'model_name': model.name,
        'quantized_layers': [layer.__class__.__name__ for layer in qmodel.layers],
        'layer_audit': [],
    }

    for layer in qmodel.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        entry = {
            'layer_name': layer.name,
            'layer_type': layer.__class__.__name__,
            'weights': [],
        }
        for idx, tensor in enumerate(weights):
            tensor_summary = summarize_tensor(tensor)
            tensor_summary['index'] = idx
            tensor_summary['shape'] = list(np.asarray(tensor).shape)
            entry['weights'].append(tensor_summary)
        report['layer_audit'].append(entry)

    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
