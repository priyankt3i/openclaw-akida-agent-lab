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
ACTIVATION_BITS = 4
ARTIFACT_DIR = Path('prototype-1/artifacts/kernel4_single_block_calibration_check')
OUT_JSON = ARTIFACT_DIR / 'summary.json'
OUT_NPZ = ARTIFACT_DIR / 'arrays.npz'


def rel_mse(reference, candidate):
    diff = candidate - reference
    return float(np.mean(diff ** 2) / (np.mean(reference ** 2) + 1e-12))


def signed_impulse_fidelity(reference, candidate):
    active = np.abs(reference) > 1e-6
    if not np.any(active):
        return 0.0
    sign_match = np.mean((np.sign(reference[active]) == np.sign(candidate[active])).astype(np.float32))
    mag_ratio = np.mean(np.abs(candidate[active]) / (np.abs(reference[active]) + 1e-6))
    return float(sign_match * mag_ratio)


def nonzero_fraction(x):
    return float(np.mean(np.abs(x) > 1e-9))


def best_scalar_align(reference, candidate):
    denom = float(np.sum(candidate * candidate))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(reference * candidate) / denom)


def build_model():
    x_in = layers.Input(batch_shape=(1, 1, 1, EXTERNAL_WIDTH), name='frame')
    x = layers.Conv2D(BOTTLENECK_WIDTH, kernel_size=1, use_bias=False, name='proj_in')(x_in)
    x = DepthwiseBufferTempConv(kernel_size=KERNEL_SIZE, name='dw_btc4')(x)
    x = layers.Conv2D(EXTERNAL_WIDTH, kernel_size=1, use_bias=False, name='proj_out')(x)
    model = Model(x_in, x, name='kernel4_single_block_calibration_check')
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
            kernel = np.ones((KERNEL_SIZE, layer.weights[0].shape[1]), dtype=np.float32)
            layer.set_weights([kernel])
    return model


def build_sequence(amplitude=AMPLITUDE):
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


def make_calibrations(seq):
    base = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, -2, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-2, 2, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, -3, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-3, 3, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, -4, 2, -2, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-4, 4, -2, 2, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [6, -6, 3, -3, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-6, 6, -3, 3, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, -8, 4, -4, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-8, 8, -4, 4, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)
    toy = seq.astype(np.float32)
    clipped = np.clip(toy, -6.0, 6.0)
    wide = np.concatenate([base * 2.0, toy, -toy], axis=0)
    toy_repeat = np.concatenate([toy, toy, -0.5 * toy], axis=0)
    return [
        {'name': 'baseline_minmax_scale1', 'samples': base.reshape(-1, 1, 1, EXTERNAL_WIDTH), 'epochs': 1},
        {'name': 'baseline_minmax_scale2', 'samples': (2.0 * base).reshape(-1, 1, 1, EXTERNAL_WIDTH), 'epochs': 1},
        {'name': 'toy_task_exact', 'samples': toy.reshape(-1, 1, 1, EXTERNAL_WIDTH), 'epochs': 1},
        {'name': 'percentile_like_clipped_toy', 'samples': clipped.reshape(-1, 1, 1, EXTERNAL_WIDTH), 'epochs': 1},
        {'name': 'wide_tail_mix', 'samples': wide.reshape(-1, 1, 1, EXTERNAL_WIDTH), 'epochs': 1},
        {'name': 'toy_task_exact_epochs3', 'samples': toy_repeat.reshape(-1, 1, 1, EXTERNAL_WIDTH), 'epochs': 3},
    ]


def evaluate_variant(variant, seq):
    model = build_model()
    qparams = QuantizationParams(input_dtype='int8', activation_bits=ACTIVATION_BITS, weight_bits=4, input_weight_bits=8)
    qmodel = quantize(model, qparams=qparams, samples=variant['samples'], batch_size=1, epochs=variant['epochs'])
    ak_model = convert(qmodel)
    q_out = run_tf_stream(qmodel, seq)
    ak_out = run_akida_stream(ak_model, seq)
    scalar = best_scalar_align(q_out, ak_out)
    ak_out_scalar = ak_out * scalar
    return {
        'name': variant['name'],
        'epochs': variant['epochs'],
        'num_samples': int(variant['samples'].shape[0]),
        'sample_abs_max': float(np.max(np.abs(variant['samples']))),
        'metrics': {
            'raw_rel_mse': rel_mse(q_out, ak_out),
            'scalar_rel_mse': rel_mse(q_out, ak_out_scalar),
            'raw_signed_impulse_fidelity': signed_impulse_fidelity(q_out, ak_out),
            'scalar_signed_impulse_fidelity': signed_impulse_fidelity(q_out, ak_out_scalar),
            'scalar_align_scale': scalar,
            'akida_nonzero_fraction': nonzero_fraction(ak_out),
            'q_nonzero_fraction': nonzero_fraction(q_out),
            'akida_abs_sum': float(np.sum(np.abs(ak_out))),
            'q_abs_sum': float(np.sum(np.abs(q_out))),
            'compression_ratio_abs_sum': float(np.sum(np.abs(ak_out)) / (np.sum(np.abs(q_out)) + 1e-12)),
            'all_zero_akida': bool(np.all(np.abs(ak_out) <= 1e-9)),
        },
        'arrays': {
            'q_out': q_out,
            'ak_out': ak_out,
            'ak_out_scalar': ak_out_scalar,
        },
    }


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    seq = build_sequence()
    variants = make_calibrations(seq)
    records = []
    for variant in variants:
        print(f"evaluating {variant['name']}", flush=True)
        records.append(evaluate_variant(variant, seq))

    best = min(records, key=lambda r: (
        r['metrics']['scalar_rel_mse'],
        -r['metrics']['scalar_signed_impulse_fidelity'],
        -r['metrics']['akida_nonzero_fraction'],
    ))
    baseline = next(r for r in records if r['name'] == 'baseline_minmax_scale2')

    np.savez(
        OUT_NPZ,
        input_sequence=seq[None, ...],
        best_q_out=best['arrays']['q_out'][None, ...],
        best_ak_out=best['arrays']['ak_out'][None, ...],
        best_ak_out_scalar=best['arrays']['ak_out_scalar'][None, ...],
    )

    summary = {
        'seed': SEED,
        'fixed_config': {
            'kernel_size': KERNEL_SIZE,
            'tokens': TOKENS,
            'bottleneck_width': BOTTLENECK_WIDTH,
            'amplitude': AMPLITUDE,
            'pre_gain': PRE_GAIN,
            'activation_bits': ACTIVATION_BITS,
            'kernel_name': 'baseline_ones_sum4',
        },
        'quantizeml_calibration_note': 'Local Keras path exposes min/max observer calibration with EMA. No direct MSE/percentile selector was found, so variants emulate percentile-like behavior by clipping calibration samples and by changing sample distribution / epochs.',
        'baseline_reference': {
            'source': 'prototype-1/artifacts/kernel4_single_block_kernelnorm_followup/summary.json',
            'raw_rel_mse': 0.8430492281913757,
            'scalar_rel_mse': 0.0042974576354026794,
            'scalar_align_scale': 12.166665077209473,
            'akida_nonzero_fraction': 0.125,
        },
        'best_variant': {
            'name': best['name'],
            'epochs': best['epochs'],
            'num_samples': best['num_samples'],
            'sample_abs_max': best['sample_abs_max'],
            'metrics': best['metrics'],
        },
        'baseline_variant_repeat': {
            'name': baseline['name'],
            'epochs': baseline['epochs'],
            'num_samples': baseline['num_samples'],
            'sample_abs_max': baseline['sample_abs_max'],
            'metrics': baseline['metrics'],
        },
        'best_minus_baseline_repeat': {
            'raw_rel_mse_delta': float(best['metrics']['raw_rel_mse'] - baseline['metrics']['raw_rel_mse']),
            'scalar_rel_mse_delta': float(best['metrics']['scalar_rel_mse'] - baseline['metrics']['scalar_rel_mse']),
            'scalar_align_scale_delta': float(best['metrics']['scalar_align_scale'] - baseline['metrics']['scalar_align_scale']),
            'akida_nonzero_fraction_delta': float(best['metrics']['akida_nonzero_fraction'] - baseline['metrics']['akida_nonzero_fraction']),
            'compression_ratio_delta': float(best['metrics']['compression_ratio_abs_sum'] - baseline['metrics']['compression_ratio_abs_sum']),
        },
        'records': [
            {
                'name': r['name'],
                'epochs': r['epochs'],
                'num_samples': r['num_samples'],
                'sample_abs_max': r['sample_abs_max'],
                'metrics': r['metrics'],
            }
            for r in records
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
