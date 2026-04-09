import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_keras import Sequential, layers
from quantizeml.layers import InputQuantizer, QuantizedDepthwiseBufferTempConv, QuantizedReLU
from cnn2snn import convert

SEED = 26
KERNEL_SIZE = 6
ARTIFACT_DIR = Path('prototype-1/artifacts/fusion_probe_qdwbtc_kernel6_explicit_contract')
OUT = ARTIFACT_DIR / 'result.json'

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


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    sample = np.random.normal(size=(1, 1, 1, 4)).astype('float32')
    report = {
        'seed': SEED,
        'kernel_size': KERNEL_SIZE,
        'source_guidance': {
            'file': '/usr/local/lib/python3.11/site-packages/quantizeml/layers/buffer_temp_conv.py',
            'note': 'BaseBufferTempConv._init_fifo tiles tf.expand_dims(new_sample, axis=-2) with 5 multiples, so new_sample must be rank 4 (B,H,W,C), not rank 5.'
        },
        'sample_shape': list(sample.shape),
    }

    try:
        model = Sequential([
            layers.Input(batch_shape=(1, 1, 1, 4), name='frame'),
            InputQuantizer(bitwidth=8, signed=True, name='input_quantizer'),
            QuantizedDepthwiseBufferTempConv(kernel_size=KERNEL_SIZE, quant_config=qcfg_dw, name='qdw_btc6'),
            QuantizedReLU(max_value=6.0, quant_config=qcfg_relu, name='qrelu6'),
        ], name='fusion_probe_qdwbtc_kernel6_explicit_contract')
        warmup = np.asarray(model(sample))
        report['quantized_call'] = {
            'status': 'ok',
            'output_shape': list(warmup.shape),
            'layers': [layer.__class__.__name__ for layer in model.layers],
        }
    except Exception as exc:
        report['quantized_call'] = {
            'status': 'error',
            'error_type': type(exc).__name__,
            'message': str(exc),
        }
        OUT.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

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
