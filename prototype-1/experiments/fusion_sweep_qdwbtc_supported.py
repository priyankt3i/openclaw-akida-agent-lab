import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_keras import Sequential, layers
from quantizeml.layers import InputQuantizer, QuantizedDepthwiseBufferTempConv, QuantizedReLU
from cnn2snn import convert

SEED = 21
KERNEL_SIZES = [2, 4, 6, 8, 10]
ARTIFACT_DIR = Path('prototype-1/artifacts/fusion_sweep_qdwbtc_supported')
OUT = ARTIFACT_DIR / 'summary.json'

np.random.seed(SEED)
tf.random.set_seed(SEED)


qcfg_dw = {
    'weight_quantizer': {'bitwidth': 4, 'axis': None},
    'output_quantizer': {'bitwidth': 4, 'axis': 'per-tensor', 'signed': True},
    'buffer_bitwidth': 16,
}
qcfg_relu = {
    'output_quantizer': {'bitwidth': 4, 'axis': 'per-tensor', 'signed': False},
}


def run_probe(kernel_size: int):
    calibration = np.random.normal(loc=0.0, scale=1.0, size=(8, 1, 1, 1, 4)).astype('float32')
    report = {
        'seed': SEED,
        'kernel_size': kernel_size,
        'calibration_shape': list(calibration.shape),
        'probe_type': '4D streaming input, explicit quantized temporal primitive',
        'activation_quantization': 'per-tensor',
        'relu_max_value': 6.0,
        'calibration': 'random normal noise',
    }

    try:
        model = Sequential([
            layers.Input(batch_shape=(1, 1, 1, 1, 4), name='stream'),
            InputQuantizer(bitwidth=8, signed=True, name='input_quantizer'),
            QuantizedDepthwiseBufferTempConv(kernel_size=kernel_size, quant_config=qcfg_dw, name=f'qdw_btc{kernel_size}'),
            QuantizedReLU(max_value=6.0, quant_config=qcfg_relu, name='qrelu6'),
        ], name=f'fusion_probe_qdwbtc_kernel{kernel_size}')
        report['model_layers'] = [layer.__class__.__name__ for layer in model.layers]
        report['layer_names'] = [layer.name for layer in model.layers]
        calib_out = np.asarray(model(calibration[:1]))
        report['quantize_stage'] = {
            'status': 'ok',
            'warmup_output_shape': list(calib_out.shape),
            'evidence': f'explicit quantized model executed on calibration sample with output shape {list(calib_out.shape)}',
        }
    except Exception as exc:
        report['quantize_stage'] = {
            'status': 'error',
            'error_type': type(exc).__name__,
            'message': str(exc),
        }
        return report

    try:
        ak_model = convert(model)
        akida_model_str = str(ak_model)
        akida_layers = [layer.__class__.__name__ for layer in ak_model.layers]
        akida_layer_names = [getattr(layer, 'name', None) for layer in ak_model.layers]
        fused_temporal = any('temp' in (name or '').lower() or 'buffer' in (name or '').lower() or 'temporal' in (name or '').lower() for name in akida_layer_names) or any('temp' in cls.lower() or 'buffer' in cls.lower() or 'temporal' in cls.lower() for cls in akida_layers)
        report['convert_stage'] = {
            'status': 'ok',
            'akida_model_type': type(ak_model).__name__,
            'akida_layers': akida_layers,
            'akida_layer_names': akida_layer_names,
            'fused_temporal_layer_in_summary': fused_temporal,
            'akida_model_str': akida_model_str,
        }
    except Exception as exc:
        report['convert_stage'] = {
            'status': 'error',
            'error_type': type(exc).__name__,
            'message': str(exc),
        }

    return report


if __name__ == '__main__':
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        'seed': SEED,
        'kernel_sizes': KERNEL_SIZES,
        'results': [],
    }
    for kernel_size in KERNEL_SIZES:
        report = run_probe(kernel_size)
        summary['results'].append(report)
        (ARTIFACT_DIR / f'kernel_{kernel_size}.json').write_text(json.dumps(report, indent=2))

    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
