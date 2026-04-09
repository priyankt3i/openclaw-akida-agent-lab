import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_keras import Model, layers
from quantizeml.layers import DepthwiseBufferTempConv, reset_buffers
from quantizeml.models.quantize import QuantizationParams, quantize
from cnn2snn import convert

SEED = 23
EXTERNAL_WIDTH = 16
KERNEL_SIZE = 4
TOKENS = 12
ARTIFACT_DIR = Path('prototype-1/artifacts/kernel4_double_stack_fidelity_sweep')
OUT_JSON = ARTIFACT_DIR / 'summary.json'
OUT_NPZ = ARTIFACT_DIR / 'arrays.npz'


def rel_mse(reference, candidate):
    diff = candidate - reference
    return float(np.mean(diff ** 2) / (np.mean(reference ** 2) + 1e-12))


def signed_impulse_fidelity(reference, candidate):
    active = np.abs(reference) > 1e-6
    if not np.any(active):
        return 0.0
    return float(np.mean((np.sign(reference[active]) == np.sign(candidate[active])).astype(np.float32)))


def nonzero_fraction(x):
    return float(np.mean(np.abs(x) > 1e-9))


def best_scalar_align(reference, candidate):
    denom = float(np.sum(candidate * candidate))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(reference * candidate) / denom)


def build_model(blocks, bottleneck_width, pre_gain):
    x_in = layers.Input(batch_shape=(1, 1, 1, EXTERNAL_WIDTH), name='frame')
    x = x_in
    for block_idx in range(blocks):
        x = layers.Conv2D(bottleneck_width, kernel_size=1, use_bias=False, name=f'proj_in_{block_idx + 1}')(x)
        x = DepthwiseBufferTempConv(kernel_size=KERNEL_SIZE, name=f'dw_btc4_{block_idx + 1}')(x)
        x = layers.Conv2D(EXTERNAL_WIDTH, kernel_size=1, use_bias=False, name=f'proj_out_{block_idx + 1}')(x)
    model = Model(x_in, x, name=f'kernel4_qdwbtc_{blocks}block_probe')
    _ = model(np.zeros((1, 1, 1, EXTERNAL_WIDTH), dtype=np.float32))
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            kernel = np.zeros(layer.kernel.shape, dtype=np.float32)
            out_ch = layer.filters
            in_ch = layer.kernel.shape[2]
            gain = pre_gain if 'proj_in_' in layer.name else 1.0
            for idx in range(min(in_ch, out_ch)):
                kernel[0, 0, idx, idx] = gain
            layer.set_weights([kernel])
        elif isinstance(layer, DepthwiseBufferTempConv):
            kernel = np.ones((KERNEL_SIZE, layer.weights[0].shape[1]), dtype=np.float32)
            layer.set_weights([kernel])
    return model


def build_sequence(amplitude):
    seq = np.zeros((TOKENS, EXTERNAL_WIDTH), dtype=np.float32)
    seq[0, :4] = amplitude
    seq[1, :4] = -0.5 * amplitude
    seq[2, 4:8] = amplitude
    seq[4, 8:12] = 0.75 * amplitude
    seq[6, 12:16] = -0.75 * amplitude
    seq[8, :8] = 0.5 * amplitude
    seq[10, 8:16] = amplitude
    return seq


def run_tf_stream(model, seq):
    reset_buffers(model)
    outputs = []
    for frame in seq:
        x = frame.reshape(1, 1, 1, -1).astype(np.float32)
        outputs.append(np.asarray(model(x, training=False)).reshape(-1))
    return np.stack(outputs, axis=0)


def run_akida_stream(model, seq):
    outputs = []
    for frame in seq:
        x = np.clip(np.rint(frame), -128, 127).reshape(1, 1, 1, -1).astype(np.int8)
        outputs.append(np.asarray(model.predict(x)).reshape(-1))
    return np.stack(outputs, axis=0)


def summarize_metrics(q_out, ak_out_raw):
    scalar = best_scalar_align(q_out, ak_out_raw)
    ak_out_scalar = ak_out_raw * scalar
    return {
        'raw_rel_mse': rel_mse(q_out, ak_out_raw),
        'scalar_rel_mse': rel_mse(q_out, ak_out_scalar),
        'raw_signed_impulse_fidelity': signed_impulse_fidelity(q_out, ak_out_raw),
        'scalar_signed_impulse_fidelity': signed_impulse_fidelity(q_out, ak_out_scalar),
        'scalar_align_scale': scalar,
        'akida_nonzero_fraction': nonzero_fraction(ak_out_raw),
        'q_nonzero_fraction': nonzero_fraction(q_out),
        'all_zero_akida': bool(np.all(np.abs(ak_out_raw) <= 1e-9)),
        'akida_abs_sum': float(np.sum(np.abs(ak_out_raw))),
        'q_abs_sum': float(np.sum(np.abs(q_out))),
        'ak_out_scalar': ak_out_scalar,
    }


def evaluate_config(blocks, bottleneck_width, amplitude, pre_gain, calibration):
    cfg = {
        'blocks': blocks,
        'bottleneck_width': bottleneck_width,
        'amplitude': amplitude,
        'pre_gain': pre_gain,
    }
    print(f'evaluating {cfg}', flush=True)
    seq = build_sequence(amplitude)
    model = build_model(blocks, bottleneck_width, pre_gain)
    qparams = QuantizationParams(input_dtype='int8', activation_bits=4, weight_bits=4, input_weight_bits=8)
    qmodel = quantize(model, qparams=qparams, samples=calibration, batch_size=1)
    ak_model = convert(qmodel)
    q_out = run_tf_stream(qmodel, seq)
    ak_out_raw = run_akida_stream(ak_model, seq)
    metrics = summarize_metrics(q_out, ak_out_raw)
    return cfg, q_out, ak_out_raw, metrics


def choose_best(records):
    return min(records, key=lambda r: (r['metrics']['scalar_rel_mse'], -r['metrics']['scalar_signed_impulse_fidelity'], -r['metrics']['akida_nonzero_fraction']))


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    calibration = np.random.randint(-6, 7, size=(64, 1, 1, EXTERNAL_WIDTH)).astype(np.float32)
    grid = [
        {'bottleneck_width': 4, 'amplitude': 4.0, 'pre_gain': 1.0},
        {'bottleneck_width': 4, 'amplitude': 4.0, 'pre_gain': 3.0},
        {'bottleneck_width': 4, 'amplitude': 8.0, 'pre_gain': 1.0},
        {'bottleneck_width': 4, 'amplitude': 8.0, 'pre_gain': 3.0},
        {'bottleneck_width': 8, 'amplitude': 4.0, 'pre_gain': 1.0},
        {'bottleneck_width': 8, 'amplitude': 4.0, 'pre_gain': 3.0},
        {'bottleneck_width': 8, 'amplitude': 8.0, 'pre_gain': 1.0},
        {'bottleneck_width': 8, 'amplitude': 8.0, 'pre_gain': 3.0},
    ]

    records = []
    for blocks in [1, 2]:
        for item in grid:
            cfg, q_out, ak_out_raw, metrics = evaluate_config(blocks=blocks, calibration=calibration, **item)
            records.append({'config': cfg, 'metrics': metrics, 'q_out': q_out, 'ak_out_raw': ak_out_raw})

    single = [r for r in records if r['config']['blocks'] == 1]
    double = [r for r in records if r['config']['blocks'] == 2]
    best_single = choose_best(single)
    best_double = choose_best(double)

    np.savez(
        OUT_NPZ,
        best_double_reference=best_double['q_out'][None, ...],
        best_double_candidate=best_double['metrics']['ak_out_scalar'][None, ...],
        best_single_reference=best_single['q_out'][None, ...],
        best_single_candidate=best_single['metrics']['ak_out_scalar'][None, ...],
    )

    serializable = []
    for r in records:
        serializable.append({
            'config': r['config'],
            'metrics': {k: v for k, v in r['metrics'].items() if k != 'ak_out_scalar'},
        })

    report = {
        'seed': SEED,
        'kernel_size': KERNEL_SIZE,
        'tokens': TOKENS,
        'grid_size': len(grid),
        'best_double_stack': {
            'config': best_double['config'],
            'metrics': {k: v for k, v in best_double['metrics'].items() if k != 'ak_out_scalar'},
        },
        'best_single_block': {
            'config': best_single['config'],
            'metrics': {k: v for k, v in best_single['metrics'].items() if k != 'ak_out_scalar'},
        },
        'stack_minus_single': {
            'scalar_rel_mse_delta': float(best_double['metrics']['scalar_rel_mse'] - best_single['metrics']['scalar_rel_mse']),
            'scalar_signed_impulse_fidelity_delta': float(best_double['metrics']['scalar_signed_impulse_fidelity'] - best_single['metrics']['scalar_signed_impulse_fidelity']),
            'akida_nonzero_fraction_delta': float(best_double['metrics']['akida_nonzero_fraction'] - best_single['metrics']['akida_nonzero_fraction']),
        },
        'records': serializable,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
