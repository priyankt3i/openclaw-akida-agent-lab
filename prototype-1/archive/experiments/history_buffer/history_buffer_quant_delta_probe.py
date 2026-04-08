import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from cnn2snn import quantize, convert, check_model_compatibility
from tf_keras import Sequential, layers

SEED = 23
BATCH = 8
TOKENS = 256
DIM = 16
MODEL_HISTORY = 6
REPAIRED_HISTORY = 4
HIDDEN_1 = 64
HIDDEN_2 = 32
OUT_DIM = 16
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT = ARTIFACT_DIR / 'history_buffer_quant_delta_probe.json'


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


def rel_mse(a, b):
    return float(np.mean((a - b) ** 2) / (np.mean(a ** 2) + 1e-12))


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def summarize_error(reference, candidate):
    err = candidate - reference
    per_token_rel_mse = np.mean(err ** 2, axis=(0, 2)) / (np.mean(reference ** 2, axis=(0, 2)) + 1e-12)
    signed_mean_per_dim = np.mean(err, axis=(0, 1))
    signed_mean_per_token = np.mean(err, axis=(0, 2))
    return {
        'global_rel_mse': rel_mse(reference, candidate),
        'global_rel_l2': rel_l2(reference, candidate),
        'mean_signed_error': float(np.mean(err)),
        'mean_abs_error': float(np.mean(np.abs(err))),
        'per_token_rel_mse_mean': float(np.mean(per_token_rel_mse)),
        'per_token_rel_mse_max': float(np.max(per_token_rel_mse)),
        'per_token_rel_mse_tail_mean': float(np.mean(per_token_rel_mse[-TOKENS // 4 :])),
        'signed_mean_error_per_dim': signed_mean_per_dim.tolist(),
        'signed_mean_error_per_token_head': signed_mean_per_token[:16].tolist(),
        'signed_mean_error_per_token_tail': signed_mean_per_token[-16:].tolist(),
    }


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    x = make_sequence()
    full_windows = causal_windows(x, MODEL_HISTORY)
    repaired_windows = keep_recent(full_windows, MODEL_HISTORY, REPAIRED_HISTORY, DIM)
    flat_repaired = repaired_windows.reshape(BATCH * TOKENS, MODEL_HISTORY * DIM)

    model = build_model()
    float_out = model(flat_repaired, training=False).numpy().reshape(BATCH, TOKENS, OUT_DIM)

    qmodel = quantize(model, input_weight_quantization=8, weight_quantization=4, activ_quantization=4)
    quant_out = qmodel(flat_repaired, training=False).numpy().reshape(BATCH, TOKENS, OUT_DIM)

    mean_bias = np.mean(quant_out - float_out, axis=(0, 1), keepdims=True)
    bias_corrected_out = quant_out - mean_bias

    report = {
        'seed': SEED,
        'tokens': TOKENS,
        'repaired_history': REPAIRED_HISTORY,
        'compatibility': {},
        'conversion': {},
        'float_vs_quantized_repaired': summarize_error(float_out, quant_out),
        'float_vs_bias_corrected_quantized_repaired': summarize_error(float_out, bias_corrected_out),
        'bias_correction': {
            'mean_bias_per_dim': mean_bias.reshape(-1).tolist(),
            'global_rel_mse_reduction_pct': float(
                100.0 * (
                    rel_mse(float_out, quant_out) - rel_mse(float_out, bias_corrected_out)
                ) / max(rel_mse(float_out, quant_out), 1e-12)
            ),
        },
    }

    for input_dtype in ('int8', 'uint8'):
        try:
            check_model_compatibility(model, input_dtype=input_dtype)
            report['compatibility'][input_dtype] = {'status': 'compatible'}
        except Exception as exc:
            report['compatibility'][input_dtype] = {
                'status': 'blocked',
                'error_type': type(exc).__name__,
                'message': str(exc),
            }

    try:
        ak_model = convert(qmodel)
        report['conversion'] = {
            'status': 'converted',
            'akida_model': str(ak_model),
        }
    except Exception as exc:
        report['conversion'] = {
            'status': 'blocked',
            'error_type': type(exc).__name__,
            'message': str(exc),
        }

    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
