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
BOTTLENECK_WIDTH = 4
AMPLITUDE = 8.0
PRE_GAIN = 1.0
ARTIFACT_DIR = Path('prototype-1/artifacts/kernel4_single_block_kernelnorm_followup')
OUT_JSON = ARTIFACT_DIR / 'summary.json'
OUT_NPZ = ARTIFACT_DIR / 'arrays.npz'


KERNEL_CANDIDATES = [
    {'name': 'baseline_ones_sum4', 'weights': [1.0, 1.0, 1.0, 1.0]},
    {'name': 'flat_sum1.0', 'weights': [0.25, 0.25, 0.25, 0.25]},
    {'name': 'flat_sum1.1', 'weights': [0.275, 0.275, 0.275, 0.275]},
    {'name': 'decay_sum1.0', 'weights': [0.4, 0.3, 0.2, 0.1]},
    {'name': 'decay_sum1.1', 'weights': [0.44, 0.33, 0.22, 0.11]},
    {'name': 'tail_heavy_sum1.0', 'weights': [0.1, 0.2, 0.3, 0.4]},
]


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


def build_model(kernel_weights):
    x_in = layers.Input(batch_shape=(1, 1, 1, EXTERNAL_WIDTH), name='frame')
    x = layers.Conv2D(BOTTLENECK_WIDTH, kernel_size=1, use_bias=False, name='proj_in')(x_in)
    x = DepthwiseBufferTempConv(kernel_size=KERNEL_SIZE, name='dw_btc4')(x)
    x = layers.Conv2D(EXTERNAL_WIDTH, kernel_size=1, use_bias=False, name='proj_out')(x)
    model = Model(x_in, x, name='kernel4_single_block_kernelnorm_followup')
    _ = model(np.zeros((1, 1, 1, EXTERNAL_WIDTH), dtype=np.float32))
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            kernel = np.zeros(layer.kernel.shape, dtype=np.float32)
            out_ch = layer.filters
            in_ch = layer.kernel.shape[2]
            gain = PRE_GAIN if layer.name == 'proj_in' else 1.0
            for idx in range(min(in_ch, out_ch)):
                kernel[0, 0, idx, idx] = gain
            layer.set_weights([kernel])
        elif isinstance(layer, DepthwiseBufferTempConv):
            kernel = np.tile(np.asarray(kernel_weights, dtype=np.float32)[:, None], (1, layer.weights[0].shape[1]))
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


def evaluate_kernel(kernel_name, kernel_weights, calibration, seq):
    model = build_model(kernel_weights)
    qparams = QuantizationParams(input_dtype='int8', activation_bits=4, weight_bits=4, input_weight_bits=8)
    qmodel = quantize(model, qparams=qparams, samples=calibration, batch_size=1)
    ak_model = convert(qmodel)
    q_out = run_tf_stream(qmodel, seq)
    ak_out_raw = run_akida_stream(ak_model, seq)
    metrics = summarize_metrics(q_out, ak_out_raw)
    return {
        'kernel_name': kernel_name,
        'kernel_weights': [float(x) for x in kernel_weights],
        'kernel_abs_sum': float(np.sum(np.abs(kernel_weights))),
        'metrics': metrics,
        'q_out': q_out,
        'ak_out_raw': ak_out_raw,
    }


def choose_best(records):
    return min(records, key=lambda r: (
        r['metrics']['scalar_rel_mse'],
        -r['metrics']['scalar_signed_impulse_fidelity'],
        -r['metrics']['akida_nonzero_fraction'],
    ))


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    calibration = np.random.randint(-6, 7, size=(64, 1, 1, EXTERNAL_WIDTH)).astype(np.float32)
    seq = build_sequence(AMPLITUDE)
    records = []
    for item in KERNEL_CANDIDATES:
        print(f"evaluating {item['name']} {item['weights']}", flush=True)
        records.append(evaluate_kernel(item['name'], item['weights'], calibration, seq))

    best = choose_best(records)
    np.savez(
        OUT_NPZ,
        reference=best['q_out'][None, ...],
        candidate=best['metrics']['ak_out_scalar'][None, ...],
        candidate_raw=best['ak_out_raw'][None, ...],
        input_sequence=seq[None, ...],
    )

    best_baseline = next(r for r in records if r['kernel_name'] == 'baseline_ones_sum4')
    summary = {
        'seed': SEED,
        'kernel_size': KERNEL_SIZE,
        'tokens': TOKENS,
        'fixed_config': {
            'blocks': 1,
            'bottleneck_width': BOTTLENECK_WIDTH,
            'amplitude': AMPLITUDE,
            'pre_gain': PRE_GAIN,
            'activation_bits': 4,
        },
        'best_previous_single_block_reference': {
            'source': 'prototype-1/artifacts/kernel4_double_stack_fidelity_sweep/summary.json',
            'config': {'blocks': 1, 'bottleneck_width': 4, 'amplitude': 8.0, 'pre_gain': 1.0, 'kernel_name': 'baseline_ones_sum4'},
            'metrics': {
                'scalar_rel_mse': 0.0042974576354026794,
                'scalar_signed_impulse_fidelity': 0.6666666865348816,
                'akida_nonzero_fraction': 0.125,
                'all_zero_akida': False,
            },
        },
        'best_followup': {
            'kernel_name': best['kernel_name'],
            'kernel_weights': best['kernel_weights'],
            'kernel_abs_sum': best['kernel_abs_sum'],
            'metrics': {k: v for k, v in best['metrics'].items() if k != 'ak_out_scalar'},
        },
        'baseline_repeat': {
            'kernel_name': best_baseline['kernel_name'],
            'kernel_weights': best_baseline['kernel_weights'],
            'kernel_abs_sum': best_baseline['kernel_abs_sum'],
            'metrics': {k: v for k, v in best_baseline['metrics'].items() if k != 'ak_out_scalar'},
        },
        'normalized_vs_baseline_delta': {
            'best_followup_minus_baseline_scalar_rel_mse': float(best['metrics']['scalar_rel_mse'] - best_baseline['metrics']['scalar_rel_mse']),
            'best_followup_minus_baseline_scalar_signed_impulse_fidelity': float(best['metrics']['scalar_signed_impulse_fidelity'] - best_baseline['metrics']['scalar_signed_impulse_fidelity']),
            'best_followup_minus_baseline_nonzero_fraction': float(best['metrics']['akida_nonzero_fraction'] - best_baseline['metrics']['akida_nonzero_fraction']),
        },
        'normalized_vs_previous_best_delta': {
            'scalar_rel_mse_delta': float(best['metrics']['scalar_rel_mse'] - 0.0042974576354026794),
            'scalar_signed_impulse_fidelity_delta': float(best['metrics']['scalar_signed_impulse_fidelity'] - 0.6666666865348816),
            'akida_nonzero_fraction_delta': float(best['metrics']['akida_nonzero_fraction'] - 0.125),
        },
        'records': [
            {
                'kernel_name': r['kernel_name'],
                'kernel_weights': r['kernel_weights'],
                'kernel_abs_sum': r['kernel_abs_sum'],
                'metrics': {k: v for k, v in r['metrics'].items() if k != 'ak_out_scalar'},
            }
            for r in records
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
