import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from cnn2snn import quantize, convert
from tf_keras import Sequential, layers

SEED = 23
BATCH = 4
TOKENS = 64
DIM = 16
MODEL_HISTORY = 6
REPAIRED_HISTORY = 4
HIDDEN_1 = 64
HIDDEN_2 = 32
OUT_DIM = 16
SHIFT = 7.0
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT = ARTIFACT_DIR / 'history_buffer_offset_push_probe.json'


def make_sequence(batch=BATCH, tokens=TOKENS, dim=DIM):
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


def rel_mse(ref, cand):
    return float(np.mean((cand - ref) ** 2) / (np.mean(ref ** 2) + 1e-12))


def rel_l2(ref, cand):
    return float(np.linalg.norm(cand - ref) / (np.linalg.norm(ref) + 1e-12))


def summarize(ref, cand):
    err = cand - ref
    return {
        'global_rel_mse': rel_mse(ref, cand),
        'global_rel_l2': rel_l2(ref, cand),
        'mean_abs_error': float(np.mean(np.abs(err))),
        'mean_signed_error': float(np.mean(err)),
        'nonzero_fraction': float(np.mean(cand != 0.0)),
        'output_mean': float(np.mean(cand)),
        'output_std': float(np.std(cand)),
    }


def pack_zero_clip(x_float):
    return np.clip(np.round(x_float), 0, 15).astype(np.uint8)


def pack_offset_push(x_float, shift=SHIFT):
    return np.clip(np.round(x_float + shift), 0, 15).astype(np.uint8)


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()
    qmodel = quantize(model, input_weight_quantization=8, weight_quantization=4, activ_quantization=4)
    ak_model = convert(qmodel)

    x = make_sequence()
    full_windows = causal_windows(x, MODEL_HISTORY)
    repaired_windows = keep_recent(full_windows, MODEL_HISTORY, REPAIRED_HISTORY, DIM)
    flat = repaired_windows.reshape(BATCH * TOKENS, MODEL_HISTORY * DIM)

    q_out = np.asarray(qmodel(flat, training=False))

    zero_clip_in = pack_zero_clip(flat)
    offset_push_in = pack_offset_push(flat)

    zero_clip_out = np.asarray(ak_model.predict(zero_clip_in))
    offset_push_out = np.asarray(ak_model.predict(offset_push_in))

    report = {
        'tokens': TOKENS,
        'shift': SHIFT,
        'input_stats': {
            'float_min': float(flat.min()),
            'float_max': float(flat.max()),
            'float_mean': float(flat.mean()),
            'float_std': float(flat.std()),
            'zero_clip_input_sparsity': float(1.0 - np.count_nonzero(zero_clip_in) / zero_clip_in.size),
            'offset_push_input_sparsity': float(1.0 - np.count_nonzero(offset_push_in) / offset_push_in.size),
            'zero_clip_minmax': [int(zero_clip_in.min()), int(zero_clip_in.max())],
            'offset_push_minmax': [int(offset_push_in.min()), int(offset_push_in.max())],
        },
        'baseline_zero_clip_vs_qmodel': summarize(q_out, zero_clip_out),
        'offset_push_vs_qmodel': summarize(q_out, offset_push_out),
        'offset_push_gain': {
            'rel_mse_reduction_pct': float(
                100.0 * (
                    rel_mse(q_out, zero_clip_out) - rel_mse(q_out, offset_push_out)
                ) / max(rel_mse(q_out, zero_clip_out), 1e-12)
            ),
            'rel_l2_reduction_pct': float(
                100.0 * (
                    rel_l2(q_out, zero_clip_out) - rel_l2(q_out, offset_push_out)
                ) / max(rel_l2(q_out, zero_clip_out), 1e-12)
            ),
        },
    }

    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
