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
RECOVERY_SCALES = [16.0, 18.0, 19.0, 20.0, 20.33, 21.0, 22.0, 24.0]
ARTIFACT_DIR = Path('prototype-1/artifacts/kernel4_fixed_recovery_block_test')
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


def build_model(blocks, recovery_scale):
    x_in = layers.Input(batch_shape=(1, 1, 1, EXTERNAL_WIDTH), name='frame')
    x = x_in
    for block_idx in range(blocks):
        x = layers.Conv2D(BOTTLENECK_WIDTH, kernel_size=1, use_bias=False, name=f'proj_in_{block_idx + 1}')(x)
        x = DepthwiseBufferTempConv(kernel_size=KERNEL_SIZE, name=f'dw_btc4_{block_idx + 1}')(x)
        x = layers.Conv2D(EXTERNAL_WIDTH, kernel_size=1, use_bias=False, name=f'proj_out_{block_idx + 1}')(x)
        x = layers.Conv2D(EXTERNAL_WIDTH, kernel_size=1, use_bias=False, name=f'recovery_{block_idx + 1}')(x)
    model = Model(x_in, x, name=f'kernel4_fixed_recovery_{blocks}block')
    _ = model(np.zeros((1, 1, 1, EXTERNAL_WIDTH), dtype=np.float32))
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            kernel = np.zeros(layer.kernel.shape, dtype=np.float32)
            out_ch = layer.filters
            in_ch = layer.kernel.shape[2]
            if layer.name.startswith('proj_in_'):
                gain = PRE_GAIN
            elif layer.name.startswith('recovery_'):
                gain = recovery_scale
            else:
                gain = 1.0
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


def make_calibration():
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
    return (2.0 * base).reshape(-1, 1, 1, EXTERNAL_WIDTH)


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


def evaluate(blocks, recovery_scale, seq, calibration):
    model = build_model(blocks=blocks, recovery_scale=recovery_scale)
    qparams = QuantizationParams(input_dtype='int8', activation_bits=ACTIVATION_BITS, weight_bits=4, input_weight_bits=8)
    qmodel = quantize(model, qparams=qparams, samples=calibration, batch_size=1)
    ak_model = convert(qmodel)
    q_out = run_tf_stream(qmodel, seq)
    ak_out = run_akida_stream(ak_model, seq)
    return {
        'blocks': blocks,
        'recovery_scale': recovery_scale,
        'metrics': {
            'rel_mse_quant_vs_akida': rel_mse(q_out, ak_out),
            'signed_impulse_fidelity_quant_vs_akida': signed_impulse_fidelity(q_out, ak_out),
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
        },
    }


def choose_best(records):
    return min(records, key=lambda r: (
        r['metrics']['rel_mse_quant_vs_akida'],
        -r['metrics']['signed_impulse_fidelity_quant_vs_akida'],
        -r['metrics']['akida_nonzero_fraction'],
    ))


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    seq = build_sequence()
    calibration = make_calibration()
    single_records = []
    double_records = []
    for scale in RECOVERY_SCALES:
        print(f'evaluating blocks=1 recovery_scale={scale}', flush=True)
        single_records.append(evaluate(blocks=1, recovery_scale=scale, seq=seq, calibration=calibration))
        print(f'evaluating blocks=2 recovery_scale={scale}', flush=True)
        double_records.append(evaluate(blocks=2, recovery_scale=scale, seq=seq, calibration=calibration))

    best_single = choose_best(single_records)
    best_double = choose_best(double_records)

    raw_single_baseline = {
        'source': 'prototype-1/artifacts/kernel4_single_block_calibration_check/summary.json baseline_minmax_scale2',
        'rel_mse_quant_vs_akida': 0.9040588736534119,
        'signed_impulse_fidelity_quant_vs_akida': 0.04918031491068521,
        'akida_nonzero_fraction': 0.125,
        'compression_ratio_abs_sum': 0.049180314666551235,
        'scalar_align_scale': 20.33333396911621,
    }
    raw_double_baseline = {
        'source': 'prototype-1/artifacts/kernel4_double_stack_fidelity_sweep/summary.json best_double_stack',
        'rel_mse_quant_vs_akida': 0.8824626803398132,
        'signed_impulse_fidelity_quant_vs_akida': 0.25,
        'akida_nonzero_fraction': 0.0625,
        'compression_ratio_abs_sum': 0.04411765137379098,
        'scalar_align_scale': 10.999999046325684,
    }

    np.savez(
        OUT_NPZ,
        input_sequence=seq[None, ...],
        best_single_q_out=best_single['arrays']['q_out'][None, ...],
        best_single_ak_out=best_single['arrays']['ak_out'][None, ...],
        best_double_q_out=best_double['arrays']['q_out'][None, ...],
        best_double_ak_out=best_double['arrays']['ak_out'][None, ...],
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
            'calibration': 'baseline_minmax_scale2',
            'recovery_stage': 'per-block pointwise 1x1 diagonal amplifier, no bias',
        },
        'tested_scales': RECOVERY_SCALES,
        'best_single_block': {
            'blocks': best_single['blocks'],
            'recovery_scale': best_single['recovery_scale'],
            'metrics': best_single['metrics'],
            'vs_raw_single': {
                'rel_mse_delta': float(best_single['metrics']['rel_mse_quant_vs_akida'] - raw_single_baseline['rel_mse_quant_vs_akida']),
                'signed_impulse_fidelity_delta': float(best_single['metrics']['signed_impulse_fidelity_quant_vs_akida'] - raw_single_baseline['signed_impulse_fidelity_quant_vs_akida']),
                'akida_nonzero_fraction_delta': float(best_single['metrics']['akida_nonzero_fraction'] - raw_single_baseline['akida_nonzero_fraction']),
                'compression_ratio_delta': float(best_single['metrics']['compression_ratio_abs_sum'] - raw_single_baseline['compression_ratio_abs_sum']),
            },
        },
        'best_double_block': {
            'blocks': best_double['blocks'],
            'recovery_scale': best_double['recovery_scale'],
            'metrics': best_double['metrics'],
            'vs_raw_double': {
                'rel_mse_delta': float(best_double['metrics']['rel_mse_quant_vs_akida'] - raw_double_baseline['rel_mse_quant_vs_akida']),
                'signed_impulse_fidelity_delta': float(best_double['metrics']['signed_impulse_fidelity_quant_vs_akida'] - raw_double_baseline['signed_impulse_fidelity_quant_vs_akida']),
                'akida_nonzero_fraction_delta': float(best_double['metrics']['akida_nonzero_fraction'] - raw_double_baseline['akida_nonzero_fraction']),
                'compression_ratio_delta': float(best_double['metrics']['compression_ratio_abs_sum'] - raw_double_baseline['compression_ratio_abs_sum']),
            },
        },
        'stack_friendliness_proxy': {
            'raw_double_minus_raw_single_rel_mse': float(raw_double_baseline['rel_mse_quant_vs_akida'] - raw_single_baseline['rel_mse_quant_vs_akida']),
            'recovered_double_minus_recovered_single_rel_mse': float(best_double['metrics']['rel_mse_quant_vs_akida'] - best_single['metrics']['rel_mse_quant_vs_akida']),
            'raw_double_over_raw_single_rel_mse_ratio': float(raw_double_baseline['rel_mse_quant_vs_akida'] / (raw_single_baseline['rel_mse_quant_vs_akida'] + 1e-12)),
            'recovered_double_over_recovered_single_rel_mse_ratio': float(best_double['metrics']['rel_mse_quant_vs_akida'] / (best_single['metrics']['rel_mse_quant_vs_akida'] + 1e-12)),
            'supports_stack_friendlier_claim': bool(
                best_double['metrics']['rel_mse_quant_vs_akida'] < raw_double_baseline['rel_mse_quant_vs_akida']
                and (best_double['metrics']['rel_mse_quant_vs_akida'] - best_single['metrics']['rel_mse_quant_vs_akida'])
                <= (raw_double_baseline['rel_mse_quant_vs_akida'] - raw_single_baseline['rel_mse_quant_vs_akida'])
            ),
        },
        'supports_recovery_stage_hypothesis': bool(
            best_single['metrics']['rel_mse_quant_vs_akida'] < raw_single_baseline['rel_mse_quant_vs_akida'] - 0.05
            or best_double['metrics']['rel_mse_quant_vs_akida'] < raw_double_baseline['rel_mse_quant_vs_akida'] - 0.05
        ),
        'raw_single_baseline': raw_single_baseline,
        'raw_double_baseline': raw_double_baseline,
        'single_records': [
            {'blocks': r['blocks'], 'recovery_scale': r['recovery_scale'], 'metrics': r['metrics']}
            for r in single_records
        ],
        'double_records': [
            {'blocks': r['blocks'], 'recovery_scale': r['recovery_scale'], 'metrics': r['metrics']}
            for r in double_records
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
