import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_keras import Sequential, layers
from quantizeml.layers import InputQuantizer, QuantizedDepthwiseBufferTempConv, QuantizedReLU
from cnn2snn import convert

SEED = 16
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT = ARTIFACT_DIR / 'fusion_probe_qdwbtc_kernel16.json'

np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    qcfg_dw = {
        'weight_quantizer': {'bitwidth': 4, 'axis': None},
        'output_quantizer': {'bitwidth': 4, 'axis': 'per-tensor', 'signed': True},
        'buffer_bitwidth': 16,
    }
    qcfg_relu = {
        'output_quantizer': {'bitwidth': 4, 'axis': 'per-tensor', 'signed': False},
    }

    calibration = np.random.normal(loc=0.0, scale=1.0, size=(8, 1, 1, 1, 4)).astype('float32')

    model = Sequential([
        layers.Input(batch_shape=(1, 1, 1, 1, 4), name='stream'),
        InputQuantizer(bitwidth=8, signed=True, name='input_quantizer'),
        QuantizedDepthwiseBufferTempConv(kernel_size=16, quant_config=qcfg_dw, name='qdw_btc16'),
        QuantizedReLU(max_value=6.0, quant_config=qcfg_relu, name='qrelu6'),
    ], name='fusion_probe_qdwbtc_kernel16')

    calib_out = np.asarray(model(calibration[:1]))

    report = {
        'seed': SEED,
        'calibration_shape': list(calibration.shape),
        'model_layers': [layer.__class__.__name__ for layer in model.layers],
        'kernel_size': 16,
        'stride': 1,
        'dilation_rate': 1,
        'activation_quantization': 'per-tensor',
        'relu_max_value': 6.0,
        'warmup_output_shape': list(calib_out.shape),
    }

    try:
        ak_model = convert(model)
        report['convert'] = {
            'status': 'ok',
            'akida_model_type': type(ak_model).__name__,
            'akida_layers': [layer.__class__.__name__ for layer in ak_model.layers],
            'akida_layer_names': [getattr(layer, 'name', None) for layer in ak_model.layers],
            'akida_model_str': str(ak_model),
        }
    except Exception as exc:
        report['convert'] = {
            'status': 'error',
            'error_type': type(exc).__name__,
            'message': str(exc),
        }

    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
