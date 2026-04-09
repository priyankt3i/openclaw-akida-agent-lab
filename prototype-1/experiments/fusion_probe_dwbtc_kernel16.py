import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_keras import Sequential, layers
from quantizeml.layers import DepthwiseBufferTempConv
from quantizeml.models import QuantizationParams, quantize
from cnn2snn import convert

SEED = 16
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT = ARTIFACT_DIR / 'fusion_probe_dwbtc_kernel16.json'

np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    calibration = np.random.normal(loc=0.0, scale=1.0, size=(8, 1, 1, 4)).astype('float32')

    model = Sequential([
        layers.Input(shape=(1, 1, 4), name='frame'),
        DepthwiseBufferTempConv(kernel_size=16, name='dw_btc16'),
        layers.ReLU(max_value=6.0, name='relu6'),
    ], name='fusion_probe_dwbtc_kernel16')

    warmup_out = []
    for sample in calibration:
        warmup_out.append(np.asarray(model(sample[None, ...], training=False)))
    warmup_out = np.stack(warmup_out, axis=1)

    qparams = QuantizationParams(
        input_weight_bits=8,
        weight_bits=4,
        activation_bits=4,
        per_tensor_activations=True,
        input_dtype='int8',
    )

    report = {
        'seed': SEED,
        'probe_type': '4D streaming input, per-frame temporal probe',
        'calibration_shape': list(calibration.shape),
        'model_layers': [layer.__class__.__name__ for layer in model.layers],
        'kernel_size': 16,
        'stride': 1,
        'dilation_rate': 1,
        'activation_quantization': 'per-tensor',
        'relu_max_value': 6.0,
        'warmup_output_shape': list(warmup_out.shape),
        'warmup_zero_fraction': float(np.mean(warmup_out == 0.0)),
        'quantization_params': {
            'input_weight_bits': 8,
            'weight_bits': 4,
            'activation_bits': 4,
            'per_tensor_activations': True,
            'input_dtype': 'int8',
            'batch_size': 1,
            'calibration': 'random normal noise',
        },
    }

    try:
        qmodel = quantize(model, qparams=qparams, samples=calibration, batch_size=1)
        report['quantized_layers'] = [layer.__class__.__name__ for layer in qmodel.layers]
        report['quantized_layer_names'] = [layer.name for layer in qmodel.layers]
        ak_model = convert(qmodel)
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
