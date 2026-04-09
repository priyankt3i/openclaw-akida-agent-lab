import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_keras import Sequential, layers
from quantizeml.layers import InputQuantizer, QuantizedDepthwiseBufferTempConv, QuantizedReLU
from cnn2snn import convert

SEED = 8
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT = ARTIFACT_DIR / 'fusion_probe_qdwbtc_kernel8_shape_contract.json'

np.random.seed(SEED)
tf.random.set_seed(SEED)

QCFG_DW = {
    'weight_quantizer': {'bitwidth': 4, 'axis': None},
    'output_quantizer': {'bitwidth': 4, 'axis': 'per-tensor', 'signed': True},
    'buffer_bitwidth': 16,
}
QCFG_RELU = {
    'output_quantizer': {'bitwidth': 4, 'axis': 'per-tensor', 'signed': False},
}


def build_model(input_shape, name):
    return Sequential([
        layers.Input(shape=input_shape, batch_size=1, name='input'),
        InputQuantizer(bitwidth=8, signed=True, name='input_quantizer'),
        QuantizedDepthwiseBufferTempConv(kernel_size=8, quant_config=QCFG_DW, name='qdw_btc8'),
        QuantizedReLU(max_value=6.0, quant_config=QCFG_RELU, name='qrelu4'),
    ], name=name)


def attempt(input_shape, sample_shape, tag):
    report = {
        'input_shape': list(input_shape),
        'sample_shape': list(sample_shape),
    }
    sample = np.random.normal(size=sample_shape).astype('float32')
    try:
        model = build_model(input_shape, name=f'{tag}_kernel8')
        out = np.asarray(model(sample))
        report['quantized_call'] = {
            'status': 'ok',
            'output_shape': list(out.shape),
        }
        try:
            ak_model = convert(model)
            report['convert'] = {
                'status': 'ok',
                'akida_model_type': type(ak_model).__name__,
                'akida_layers': [layer.__class__.__name__ for layer in ak_model.layers],
                'akida_layer_names': [getattr(layer, 'name', None) for layer in ak_model.layers],
            }
        except Exception as exc:
            report['convert'] = {
                'status': 'error',
                'error_type': type(exc).__name__,
                'message': str(exc),
            }
    except Exception as exc:
        report['quantized_call'] = {
            'status': 'error',
            'error_type': type(exc).__name__,
            'message': str(exc),
        }
    return report


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'seed': SEED,
        'kernel_size': 8,
        'source_guidance': {
            'file': '/usr/local/lib/python3.11/site-packages/quantizeml/layers/buffer_temp_conv.py',
            'note': 'BaseBufferTempConv._init_fifo tiles a new_sample expanded on axis -2 with multiples [1, 1, 1, kernel_size, 1], so new_sample must be rank-4 [B,H,W,C], not rank-5 [B,T,H,W,C].',
        },
        'attempts': {
            'rank5_stream_input': attempt((1, 1, 1, 4), (1, 1, 1, 1, 4), 'rank5'),
            'rank4_frame_input': attempt((1, 1, 4), (1, 1, 1, 4), 'rank4'),
        },
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
