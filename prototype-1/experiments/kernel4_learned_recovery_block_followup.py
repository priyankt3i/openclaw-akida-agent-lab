import importlib.util
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
SWEEP_SCRIPT = ROOT / 'prototype-1/experiments/kernel4_single_block_fidelity_sweep.py'
SUMMARY_PATH = ROOT / 'prototype-1/artifacts/kernel4_single_block_bitdepth_postscale_sweep/summary.json'
ARTIFACT_DIR = ROOT / 'prototype-1/artifacts/kernel4_learned_recovery_block_followup'
RESULT_PATH = ARTIFACT_DIR / 'result.json'
ARRAYS_PATH = ARTIFACT_DIR / 'arrays.npz'

RIDGE = 1e-6


def load_sweep_module():
    spec = importlib.util.spec_from_file_location('k4_sweep', SWEEP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rel_mse(reference, candidate):
    return float(np.mean((candidate - reference) ** 2) / (np.mean(reference ** 2) + 1e-12))


def best_scalar(reference, candidate):
    denom = float(np.sum(candidate ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(reference * candidate) / denom)


def fit_pointwise_matrix(features, target, ridge=RIDGE):
    xtx = features.T @ features
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float32)
    xty = features.T @ target
    return np.linalg.solve(xtx + reg, xty)


def fit_diag(features, target, ridge=RIDGE):
    numer = np.sum(features * target, axis=0)
    denom = np.sum(features * features, axis=0) + ridge
    return numer / denom


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text())
    cfg = summary['best_trial']

    mod = load_sweep_module()
    np.random.seed(mod.SEED)
    tf.random.set_seed(mod.SEED)

    qparams = mod.QuantizationParams(
        input_dtype='int8',
        activation_bits=int(cfg['activation_bits']),
        weight_bits=4,
        input_weight_bits=8,
    )
    model = mod.build_model(int(cfg['bottleneck_width']), float(cfg['pre_gain']), float(cfg['post_scale']))
    _ = model(np.zeros((1, 1, 1, mod.EXTERNAL_WIDTH), dtype=np.float32))
    mod.init_weights(model)
    calibration = mod.make_calibration(float(cfg['calibration_scale']))
    qmodel = mod.quantize(model, qparams=qparams, samples=calibration, batch_size=1)
    ak_model = mod.convert(qmodel)

    seq = mod.make_sequence(float(cfg['input_amplitude']))
    float_out = mod.run_tf_stream(model, seq)
    quant_out = mod.run_tf_stream(qmodel, seq)
    akida_out = mod.run_akida_stream(ak_model, seq)

    scalar_quant = best_scalar(quant_out, akida_out)
    scalar_float = best_scalar(float_out, akida_out)
    scalar_quant_out = akida_out * scalar_quant
    scalar_float_out = akida_out * scalar_float

    diag_quant = fit_diag(akida_out, quant_out)
    diag_quant_out = akida_out * diag_quant.reshape(1, -1)

    diag_float = fit_diag(akida_out, float_out)
    diag_float_out = akida_out * diag_float.reshape(1, -1)

    pointwise_quant = fit_pointwise_matrix(akida_out, quant_out)
    pointwise_quant_out = akida_out @ pointwise_quant

    pointwise_float = fit_pointwise_matrix(akida_out, float_out)
    pointwise_float_out = akida_out @ pointwise_float

    results = {
        'frozen_temporal_config': cfg,
        'baseline_scalar_recovery': {
            'quant_scale': scalar_quant,
            'float_scale': scalar_float,
            'quant_rel_mse': rel_mse(quant_out, scalar_quant_out),
            'float_rel_mse': rel_mse(float_out, scalar_float_out),
        },
        'diag_pointwise_recovery': {
            'target_quant_weights': diag_quant.tolist(),
            'target_float_weights': diag_float.tolist(),
            'quant_rel_mse': rel_mse(quant_out, diag_quant_out),
            'float_rel_mse': rel_mse(float_out, diag_float_out),
        },
        'full_pointwise_recovery': {
            'target_quant_weights': pointwise_quant.tolist(),
            'target_float_weights': pointwise_float.tolist(),
            'quant_rel_mse': rel_mse(quant_out, pointwise_quant_out),
            'float_rel_mse': rel_mse(float_out, pointwise_float_out),
        },
    }

    quant_candidates = [
        ('scalar', results['baseline_scalar_recovery']['quant_rel_mse']),
        ('diag', results['diag_pointwise_recovery']['quant_rel_mse']),
        ('full', results['full_pointwise_recovery']['quant_rel_mse']),
    ]
    float_candidates = [
        ('scalar', results['baseline_scalar_recovery']['float_rel_mse']),
        ('diag', results['diag_pointwise_recovery']['float_rel_mse']),
        ('full', results['full_pointwise_recovery']['float_rel_mse']),
    ]
    best_quant_name, best_quant_mse = min(quant_candidates, key=lambda x: x[1])
    best_float_name, best_float_mse = min(float_candidates, key=lambda x: x[1])

    scalar_quant_mse = results['baseline_scalar_recovery']['quant_rel_mse']
    scalar_float_mse = results['baseline_scalar_recovery']['float_rel_mse']

    results['comparison'] = {
        'best_quant_target': {
            'stage': best_quant_name,
            'rel_mse': best_quant_mse,
            'improvement_vs_scalar': float((scalar_quant_mse - best_quant_mse) / (scalar_quant_mse + 1e-12)),
        },
        'best_float_target': {
            'stage': best_float_name,
            'rel_mse': best_float_mse,
            'improvement_vs_scalar': float((scalar_float_mse - best_float_mse) / (scalar_float_mse + 1e-12)),
        },
        'material_outperformance': bool(best_quant_name != 'scalar' and (scalar_quant_mse - best_quant_mse) > 0.05),
        'recommendation': 'keep_fixed_scalar' if scalar_quant_mse - best_quant_mse <= 0.05 else 'learned_recovery_maybe_worth_it',
    }

    RESULT_PATH.write_text(json.dumps(results, indent=2))
    np.savez(
        ARRAYS_PATH,
        input_sequence=seq,
        float_out=float_out,
        quant_out=quant_out,
        akida_out=akida_out,
        scalar_quant_out=scalar_quant_out,
        scalar_float_out=scalar_float_out,
        diag_quant_out=diag_quant_out,
        diag_float_out=diag_float_out,
        pointwise_quant_out=pointwise_quant_out,
        pointwise_float_out=pointwise_float_out,
        diag_quant_weights=diag_quant,
        diag_float_weights=diag_float,
        pointwise_quant_weights=pointwise_quant,
        pointwise_float_weights=pointwise_float,
    )
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
