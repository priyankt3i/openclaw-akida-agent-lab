import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_keras import Model, layers
from quantizeml.layers import DepthwiseBufferTempConv, reset_buffers
from quantizeml.models.quantize import QuantizationParams, quantize
from cnn2snn import convert

SEED = 23
EXTERNAL_WIDTH = 8
KERNEL_SIZE = 4
ARTIFACT_DIR = Path('prototype-1/artifacts/kernel4_single_block_bitdepth_postscale_sweep')
SUMMARY_JSON = ARTIFACT_DIR / 'summary.json'
RESULTS_CSV = ARTIFACT_DIR / 'results.csv'
ARRAYS_NPZ = ARTIFACT_DIR / 'arrays.npz'
PROGRESS_JSON = ARTIFACT_DIR / 'progress.json'

BOTTLENECKS = [2]
ACTIVATION_BITS = [4, 8]
INPUT_AMPLITUDES = [8.0, 16.0]
PRE_GAINS = [1.0, 2.0]
POST_SCALES = [1.0, 2.0, 4.0, 8.0]
CALIBRATION_SCALES = [1.0, 2.0, 4.0]


def build_model(bottleneck_width, pre_gain, post_scale):
    x_in = layers.Input(batch_shape=(1, 1, 1, EXTERNAL_WIDTH), name='frame')
    x = layers.Rescaling(scale=pre_gain, name='pre_gain')(x_in)
    x = layers.Conv2D(bottleneck_width, kernel_size=1, use_bias=False, name='proj_in')(x)
    x = DepthwiseBufferTempConv(kernel_size=KERNEL_SIZE, name='dw_btc4')(x)
    x = layers.Conv2D(EXTERNAL_WIDTH, kernel_size=1, use_bias=False, name='proj_out')(x)
    x = layers.Rescaling(scale=post_scale, name='post_scale')(x)
    return Model(x_in, x, name='kernel4_single_block_fidelity')


def init_weights(model):
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            kernel = np.zeros(layer.kernel.shape, dtype=np.float32)
            in_ch = layer.kernel.shape[2]
            out_ch = layer.filters
            for idx in range(min(in_ch, out_ch)):
                kernel[0, 0, idx, idx] = 1.0
            layer.set_weights([kernel])
        elif isinstance(layer, DepthwiseBufferTempConv):
            kernel = np.ones((KERNEL_SIZE, layer.weights[0].shape[1]), dtype=np.float32)
            layer.set_weights([kernel])


def make_sequence(amplitude):
    seq = np.zeros((8, EXTERNAL_WIDTH), dtype=np.float32)
    seq[0, 0:2] = [amplitude, -amplitude]
    seq[1, 1:3] = [-0.5 * amplitude, 0.5 * amplitude]
    seq[2, 0:2] = [0.75 * amplitude, 0.25 * amplitude]
    seq[3, 2:4] = [-amplitude, amplitude]
    seq[4, 0:4] = [0.5 * amplitude, -0.25 * amplitude, 0.25 * amplitude, -0.5 * amplitude]
    return seq


def make_calibration(calibration_scale):
    base = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0, 0],
        [-1, 1, 1, -1, 0, 0, 0, 0],
        [2, -2, 1, -1, 0, 0, 0, 0],
        [-2, 2, -1, 1, 0, 0, 0, 0],
        [3, -3, 2, -2, 0, 0, 0, 0],
        [-3, 3, -2, 2, 0, 0, 0, 0],
        [4, -4, 2, -2, 1, -1, 0, 0],
        [-4, 4, -2, 2, -1, 1, 0, 0],
        [6, -6, 3, -3, 1, -1, 0, 0],
        [-6, 6, -3, 3, -1, 1, 0, 0],
        [8, -8, 4, -4, 2, -2, 0, 0],
        [-8, 8, -4, 4, -2, 2, 0, 0],
    ], dtype=np.float32)
    samples = calibration_scale * base
    return samples.reshape(samples.shape[0], 1, 1, EXTERNAL_WIDTH)


def run_tf_stream(model, seq):
    reset_buffers(model)
    outputs = []
    for frame in seq:
        x = frame.reshape(1, 1, 1, -1).astype(np.float32)
        y = np.asarray(model(x, training=False)).reshape(-1)
        outputs.append(y)
    return np.asarray(outputs, dtype=np.float32)


def run_akida_stream(model, seq):
    outputs = []
    for frame in seq:
        x = np.clip(np.round(frame), -128, 127).reshape(1, 1, 1, -1).astype(np.int8)
        y = np.asarray(model.predict(x)).reshape(-1)
        outputs.append(y)
    return np.asarray(outputs, dtype=np.float32)


def rel_mse(reference, candidate):
    return float(np.mean((candidate - reference) ** 2) / (np.mean(reference ** 2) + 1e-12))


def best_fit_scale(reference, candidate):
    denom = float(np.sum(candidate ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(reference * candidate) / denom)


def signed_impulse_fidelity(reference, candidate):
    ref_sign = np.sign(reference)
    cand_sign = np.sign(candidate)
    active = np.abs(reference) > 1e-6
    if not np.any(active):
        return 0.0
    sign_match = np.mean((ref_sign[active] == cand_sign[active]).astype(np.float32))
    mag_ratio = np.mean(np.abs(candidate[active]) / (np.abs(reference[active]) + 1e-6))
    return float(sign_match * mag_ratio)


def meaningful_nonzero_fraction(values, threshold=1e-3):
    return float(np.mean(np.abs(values) > threshold))


def config_key(row):
    return (
        int(row['activation_bits']),
        int(row['bottleneck_width']),
        float(row['input_amplitude']),
        float(row['pre_gain']),
        float(row['post_scale']),
        float(row['calibration_scale']),
    )


def write_summary(rows, best):
    summary = {
        'seed': SEED,
        'external_width': EXTERNAL_WIDTH,
        'kernel_size': KERNEL_SIZE,
        'search_space': {
            'activation_bits': ACTIVATION_BITS,
            'bottleneck_width': BOTTLENECKS,
            'input_amplitude': INPUT_AMPLITUDES,
            'pre_gain': PRE_GAINS,
            'post_scale': POST_SCALES,
            'calibration_scale': CALIBRATION_SCALES,
        },
        'num_trials': len(rows),
        'num_successful_trials': int(sum(1 for row in rows if row['toolchain_ok'])),
        'num_nonzero_akida_trials': int(sum(1 for row in rows if row.get('akida_nonzero_fraction', 0.0) > 0.0)),
        'best_trial': best,
        'top5_by_rel_mse': sorted(
            [row for row in rows if row['toolchain_ok']],
            key=lambda row: (
                row['aligned_rel_mse_float_vs_akida'],
                row['rel_mse_float_vs_akida'],
                -row['signed_impulse_fidelity_float_vs_akida'],
                -row['akida_nonzero_fraction'],
            ),
        )[:5],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    return summary


def write_progress(rows, completed, best):
    total_trials = len(ACTIVATION_BITS) * len(BOTTLENECKS) * len(INPUT_AMPLITUDES) * len(PRE_GAINS) * len(POST_SCALES) * len(CALIBRATION_SCALES)
    progress = {
        'completed_trials': len(completed),
        'total_trials': total_trials,
        'remaining_trials': total_trials - len(completed),
        'best_trial': best,
    }
    PROGRESS_JSON.write_text(json.dumps(progress, indent=2))
    write_summary(rows, best)


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'activation_bits', 'bottleneck_width', 'input_amplitude', 'pre_gain', 'post_scale', 'calibration_scale',
        'toolchain_ok', 'error', 'rel_mse_float_vs_akida', 'rel_mse_quant_vs_akida', 'aligned_rel_mse_float_vs_akida',
        'best_fit_scale_float_from_akida', 'signed_impulse_fidelity_float_vs_akida', 'signed_impulse_fidelity_quant_vs_akida',
        'akida_nonzero_fraction', 'akida_abs_sum', 'akida_max_abs', 'float_abs_sum', 'quant_abs_sum',
        'compression_ratio_abs_sum', 'all_zero_akida'
    ]
    rows = []
    completed = set()
    if RESULTS_CSV.exists():
        with RESULTS_CSV.open(newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = dict(row)
                for key in ['activation_bits', 'bottleneck_width']:
                    parsed[key] = int(parsed[key])
                for key in ['input_amplitude', 'pre_gain', 'post_scale', 'calibration_scale', 'rel_mse_float_vs_akida', 'rel_mse_quant_vs_akida', 'aligned_rel_mse_float_vs_akida', 'best_fit_scale_float_from_akida', 'signed_impulse_fidelity_float_vs_akida', 'signed_impulse_fidelity_quant_vs_akida', 'akida_nonzero_fraction', 'akida_abs_sum', 'akida_max_abs', 'float_abs_sum', 'quant_abs_sum', 'compression_ratio_abs_sum']:
                    if parsed.get(key) in ('', 'None', None):
                        parsed[key] = None
                    else:
                        parsed[key] = float(parsed[key])
                for key in ['toolchain_ok', 'all_zero_akida']:
                    parsed[key] = str(parsed[key]).lower() == 'true'
                rows.append(parsed)
                completed.add(config_key(parsed))

    best = None
    best_key = None
    best_arrays = None
    for row in rows:
        if row['toolchain_ok']:
            score = (
                row['aligned_rel_mse_float_vs_akida'],
                row['rel_mse_float_vs_akida'],
                -row['signed_impulse_fidelity_float_vs_akida'],
                -row['akida_nonzero_fraction'],
                -row['akida_abs_sum'],
            )
            if best is None or score < best_key:
                best = row
                best_key = score

    if not RESULTS_CSV.exists():
        with RESULTS_CSV.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    base_configs = product(ACTIVATION_BITS, BOTTLENECKS, PRE_GAINS, POST_SCALES, CALIBRATION_SCALES)
    for activation_bits, bottleneck_width, pre_gain, post_scale, calibration_scale in base_configs:
        base_config = {
            'activation_bits': activation_bits,
            'bottleneck_width': bottleneck_width,
            'pre_gain': pre_gain,
            'post_scale': post_scale,
            'calibration_scale': calibration_scale,
        }
        try:
            qparams = QuantizationParams(
                input_dtype='int8',
                activation_bits=activation_bits,
                weight_bits=4,
                input_weight_bits=8,
            )
            model = build_model(bottleneck_width, pre_gain, post_scale)
            _ = model(np.zeros((1, 1, 1, EXTERNAL_WIDTH), dtype=np.float32))
            init_weights(model)
            calibration = make_calibration(calibration_scale)
            qmodel = quantize(model, qparams=qparams, samples=calibration, batch_size=1)
            ak_model = convert(qmodel)
            base_error = None
        except Exception as exc:
            model = None
            qmodel = None
            ak_model = None
            base_error = f'{type(exc).__name__}: {exc}'

        for amplitude in INPUT_AMPLITUDES:
            config = {**base_config, 'input_amplitude': amplitude}
            if config_key(config) in completed:
                continue
            try:
                if base_error is not None:
                    raise RuntimeError(base_error)

                seq = make_sequence(amplitude)
                float_out = run_tf_stream(model, seq)
                quant_out = run_tf_stream(qmodel, seq)
                akida_out = run_akida_stream(ak_model, seq)

                float_b = float_out.reshape(1, float_out.shape[0], float_out.shape[1])
                akida_b = akida_out.reshape(1, akida_out.shape[0], akida_out.shape[1])
                quant_b = quant_out.reshape(1, quant_out.shape[0], quant_out.shape[1])

                align_scale = best_fit_scale(float_out, akida_out)
                aligned_akida_out = akida_out * align_scale

                row = {
                    **config,
                    'toolchain_ok': True,
                    'rel_mse_float_vs_akida': rel_mse(float_b, akida_b),
                    'rel_mse_quant_vs_akida': rel_mse(quant_b, akida_b),
                    'aligned_rel_mse_float_vs_akida': rel_mse(float_out, aligned_akida_out),
                    'best_fit_scale_float_from_akida': align_scale,
                    'signed_impulse_fidelity_float_vs_akida': signed_impulse_fidelity(float_out, akida_out),
                    'signed_impulse_fidelity_quant_vs_akida': signed_impulse_fidelity(quant_out, akida_out),
                    'akida_nonzero_fraction': meaningful_nonzero_fraction(akida_out),
                    'akida_abs_sum': float(np.sum(np.abs(akida_out))),
                    'akida_max_abs': float(np.max(np.abs(akida_out))),
                    'float_abs_sum': float(np.sum(np.abs(float_out))),
                    'quant_abs_sum': float(np.sum(np.abs(quant_out))),
                    'compression_ratio_abs_sum': float(np.sum(np.abs(akida_out)) / (np.sum(np.abs(float_out)) + 1e-12)),
                    'all_zero_akida': bool(np.all(np.abs(akida_out) <= 1e-6)),
                }
            except Exception as exc:
                row = {
                    **config,
                    'toolchain_ok': False,
                    'error': f'{type(exc).__name__}: {exc}',
                    'rel_mse_float_vs_akida': None,
                    'rel_mse_quant_vs_akida': None,
                    'aligned_rel_mse_float_vs_akida': None,
                    'best_fit_scale_float_from_akida': None,
                    'signed_impulse_fidelity_float_vs_akida': None,
                    'signed_impulse_fidelity_quant_vs_akida': None,
                    'akida_nonzero_fraction': 0.0,
                    'akida_abs_sum': 0.0,
                    'akida_max_abs': 0.0,
                    'float_abs_sum': None,
                    'quant_abs_sum': None,
                    'compression_ratio_abs_sum': None,
                    'all_zero_akida': True,
                }
                float_out = None
                quant_out = None
                akida_out = None
                seq = make_sequence(amplitude)

            if row['toolchain_ok']:
                score = (
                    row['aligned_rel_mse_float_vs_akida'],
                    row['rel_mse_float_vs_akida'],
                    -row['signed_impulse_fidelity_float_vs_akida'],
                    -row['akida_nonzero_fraction'],
                    -row['akida_abs_sum'],
                )
                if best is None or score < best_key:
                    best = row
                    best_key = score
                    best_arrays = {
                        'reference': float_out.reshape(1, float_out.shape[0], float_out.shape[1]),
                        'candidate_akida': akida_out.reshape(1, akida_out.shape[0], akida_out.shape[1]),
                        'candidate_quant': quant_out.reshape(1, quant_out.shape[0], quant_out.shape[1]),
                        'input_sequence': seq.reshape(1, seq.shape[0], seq.shape[1]),
                    }
            rows.append(row)
            completed.add(config_key(row))
            with RESULTS_CSV.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({name: row.get(name) for name in fieldnames})
            if row['toolchain_ok'] and best_arrays is not None:
                np.savez(ARRAYS_NPZ, **best_arrays)
            write_progress(rows, completed, best)

    if best_arrays is not None:
        np.savez(ARRAYS_NPZ, **best_arrays)

    summary = write_summary(rows, best)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
