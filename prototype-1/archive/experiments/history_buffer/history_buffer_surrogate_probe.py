import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras
from tf_keras import Sequential, layers
from cnn2snn import check_model_compatibility, convert, quantize

SEED = 23
BATCH = 8
TOKENS = 32
DIM = 16
MODEL_HISTORY = 6
REPAIRED_HISTORY = 4
CURRENT_ONLY_HISTORY = 1
HIDDEN_1 = 64
HIDDEN_2 = 32
OUT_DIM = 16
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT_JSON = ARTIFACT_DIR / 'history_buffer_surrogate_probe.json'
OUT_CSV = ARTIFACT_DIR / 'history_buffer_surrogate_per_token.csv'
OUT_NPZ = ARTIFACT_DIR / 'history_buffer_surrogate_arrays.npz'


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


def mask_to_recent_history(flat_windows, model_history, keep_recent, dim):
    masked = flat_windows.reshape(flat_windows.shape[0], flat_windows.shape[1], model_history, dim).copy()
    if keep_recent < model_history:
        masked[:, :, : model_history - keep_recent, :] = 0.0
    return masked.reshape(flat_windows.shape)


def build_model():
    model = Sequential([
        layers.Input(shape=(MODEL_HISTORY * DIM,), name='history_tokens_flat'),
        layers.Dense(HIDDEN_1, use_bias=False, name='proj_in'),
        layers.ReLU(max_value=6.0, name='gate_relu6'),
        layers.Dense(HIDDEN_2, use_bias=False, name='mix_dense'),
        layers.ReLU(max_value=6.0, name='post_relu6'),
        layers.Dense(OUT_DIM, use_bias=False, name='proj_out'),
    ], name='history_buffer_dense_relu6_surrogate')
    return model


def activation_sparsity(model, sample):
    relu_layers = [layer.output for layer in model.layers if isinstance(layer, layers.ReLU)]
    if not relu_layers:
        return {}
    probe = tf_keras.Model(model.input, relu_layers)
    outputs = probe(sample, training=False)
    if not isinstance(outputs, list):
        outputs = [outputs]
    out = {}
    relu_defs = [layer for layer in model.layers if isinstance(layer, layers.ReLU)]
    for layer, output in zip(relu_defs, outputs):
        arr = np.asarray(output)
        out[layer.name] = float(np.mean(arr == 0.0))
    return out


def classify_trend(slope, ratio):
    if ratio < 0.9 and slope < 0:
        return 'decays_with_context'
    if ratio > 1.1 and slope > 0:
        return 'grows_with_context'
    return 'roughly_flat'


def summarize(reference, candidate):
    diff = candidate - reference
    per_token_rel_mse = np.mean(diff ** 2, axis=(0, 2)) / (np.mean(reference ** 2, axis=(0, 2)) + 1e-12)
    per_token_rmse = np.sqrt(np.mean(diff ** 2, axis=(0, 2)))
    prefix_rel_mse = []
    for end in range(1, reference.shape[1] + 1):
        ref_prefix = reference[:, :end, :]
        cand_prefix = candidate[:, :end, :]
        rel = np.mean((cand_prefix - ref_prefix) ** 2) / (np.mean(ref_prefix ** 2) + 1e-12)
        prefix_rel_mse.append(float(rel))
    slope = float(np.polyfit(np.arange(reference.shape[1]), per_token_rel_mse, deg=1)[0])
    early = float(np.mean(per_token_rel_mse[: reference.shape[1] // 4]))
    late = float(np.mean(per_token_rel_mse[-reference.shape[1] // 4 :]))
    ratio = float(late / max(early, 1e-12))
    return {
        'global_mse': float(np.mean(diff ** 2)),
        'global_rel_mse': float(np.mean(diff ** 2) / (np.mean(reference ** 2) + 1e-12)),
        'global_rel_l2': float(np.linalg.norm(diff) / (np.linalg.norm(reference) + 1e-12)),
        'per_token_rel_mse': per_token_rel_mse.tolist(),
        'per_token_rmse': per_token_rmse.tolist(),
        'prefix_rel_mse': prefix_rel_mse,
        'trend': {
            'per_token_rel_mse_slope': slope,
            'early_quarter_mean_rel_mse': early,
            'late_quarter_mean_rel_mse': late,
            'late_to_early_ratio': ratio,
            'classification': classify_trend(slope, ratio),
        },
    }


def compatibility_and_conversion(model):
    compat = {}
    for input_dtype in ('int8', 'uint8'):
        try:
            check_model_compatibility(model, input_dtype=input_dtype)
            compat[input_dtype] = {'status': 'compatible'}
        except Exception as exc:
            compat[input_dtype] = {
                'status': 'blocked',
                'error_type': type(exc).__name__,
                'message': str(exc),
            }

    conversion = {}
    try:
        qmodel = quantize(model, input_weight_quantization=8, weight_quantization=4, activ_quantization=4)
        conversion['quantized_layers'] = [layer.__class__.__name__ for layer in qmodel.layers]
        try:
            ak_model = convert(qmodel)
            conversion['status'] = 'converted'
            conversion['akida_model'] = str(ak_model)
        except Exception as exc:
            conversion['status'] = 'blocked_after_quantization'
            conversion['error_type'] = type(exc).__name__
            conversion['message'] = str(exc)
    except Exception as exc:
        conversion['status'] = 'blocked'
        conversion['error_type'] = type(exc).__name__
        conversion['message'] = str(exc)

    return compat, conversion


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'token_idx',
            'current_only_rel_mse',
            'repaired_rel_mse',
            'current_only_rmse',
            'repaired_rmse',
        ])
        writer.writerows(rows)


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    x = make_sequence()
    full_windows = causal_windows(x, MODEL_HISTORY)
    repaired_windows = mask_to_recent_history(full_windows, MODEL_HISTORY, REPAIRED_HISTORY, DIM)
    current_only_windows = mask_to_recent_history(full_windows, MODEL_HISTORY, CURRENT_ONLY_HISTORY, DIM)

    flat_full = full_windows.reshape(BATCH * TOKENS, MODEL_HISTORY * DIM)
    flat_repaired = repaired_windows.reshape(BATCH * TOKENS, MODEL_HISTORY * DIM)
    flat_current = current_only_windows.reshape(BATCH * TOKENS, MODEL_HISTORY * DIM)

    model = build_model()
    ref = model(flat_full, training=False).numpy().reshape(BATCH, TOKENS, OUT_DIM)
    repaired = model(flat_repaired, training=False).numpy().reshape(BATCH, TOKENS, OUT_DIM)
    current_only = model(flat_current, training=False).numpy().reshape(BATCH, TOKENS, OUT_DIM)

    current_metrics = summarize(ref, current_only)
    repaired_metrics = summarize(ref, repaired)
    compat, conversion = compatibility_and_conversion(model)

    report = {
        'seed': SEED,
        'batch': BATCH,
        'tokens': TOKENS,
        'dim': DIM,
        'model_history': MODEL_HISTORY,
        'repaired_history': REPAIRED_HISTORY,
        'current_only_history': CURRENT_ONLY_HISTORY,
        'model_name': model.name,
        'activation_sparsity': activation_sparsity(model, flat_full),
        'current_only': current_metrics,
        'repaired_history_buffer': repaired_metrics,
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
        'compatibility': compat,
        'conversion': conversion,
        'summary': {
            'reference': 'Dense/ReLU6 surrogate with a 6-token flattened causal history window.',
            'current_only_path': 'Same surrogate with older history slots zeroed so only the newest token remains.',
            'repaired_path': 'Same surrogate with a 4-token history buffer retained, testing whether a small temporal buffer repairs context drift while preserving the Akida-friendly Dense/ReLU6 structure.',
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2))
    write_csv(
        OUT_CSV,
        [
            [idx, current_metrics['per_token_rel_mse'][idx], repaired_metrics['per_token_rel_mse'][idx], current_metrics['per_token_rmse'][idx], repaired_metrics['per_token_rmse'][idx]]
            for idx in range(TOKENS)
        ],
    )
    np.savez(
        OUT_NPZ,
        reference=ref,
        repaired=repaired,
        current_only=current_only,
        full_windows=full_windows,
        repaired_windows=repaired_windows,
        current_only_windows=current_only_windows,
    )

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
