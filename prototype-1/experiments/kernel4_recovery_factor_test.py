import importlib.util
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
SWEEP_SCRIPT = ROOT / 'prototype-1/experiments/kernel4_single_block_fidelity_sweep.py'
SUMMARY_PATH = ROOT / 'prototype-1/artifacts/kernel4_single_block_bitdepth_postscale_sweep/summary.json'
ARTIFACT_DIR = ROOT / 'prototype-1/artifacts/kernel4_recovery_factor_test'
RESULT_PATH = ARTIFACT_DIR / 'result.json'
ARRAYS_PATH = ARTIFACT_DIR / 'arrays.npz'


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
    scalar_joint = best_scalar(np.concatenate([float_out.reshape(-1), quant_out.reshape(-1)]), np.concatenate([akida_out.reshape(-1), akida_out.reshape(-1)]))

    scaled_quant = akida_out * scalar_quant
    scaled_float = akida_out * scalar_float
    scaled_joint = akida_out * scalar_joint

    result = {
        'config': cfg,
        'scalar_fits': {
            'quant_reference': scalar_quant,
            'float_reference': scalar_float,
            'joint_reference': scalar_joint,
        },
        'metrics': {
            'float_vs_akida_before': rel_mse(float_out, akida_out),
            'float_vs_akida_after_float_scalar': rel_mse(float_out, scaled_float),
            'float_vs_akida_after_quant_scalar': rel_mse(float_out, scaled_quant),
            'float_vs_akida_after_joint_scalar': rel_mse(float_out, scaled_joint),
            'quant_vs_akida_before': rel_mse(quant_out, akida_out),
            'quant_vs_akida_after_quant_scalar': rel_mse(quant_out, scaled_quant),
            'quant_vs_akida_after_float_scalar': rel_mse(quant_out, scaled_float),
            'quant_vs_akida_after_joint_scalar': rel_mse(quant_out, scaled_joint),
        },
        'delta_between_optimal_scalars': abs(scalar_float - scalar_quant),
        'interpretation': {
            'shared_scalar_is_effective': bool(rel_mse(quant_out, scaled_quant) < 0.05 and rel_mse(float_out, scaled_quant) < 0.05),
            'mostly_calibratable_not_structural': bool(rel_mse(quant_out, scaled_quant) < 0.05),
        },
    }

    RESULT_PATH.write_text(json.dumps(result, indent=2))
    np.savez(
        ARRAYS_PATH,
        input_sequence=seq,
        float_out=float_out,
        quant_out=quant_out,
        akida_out=akida_out,
        akida_scaled_quant=scaled_quant,
        akida_scaled_float=scaled_float,
        akida_scaled_joint=scaled_joint,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
