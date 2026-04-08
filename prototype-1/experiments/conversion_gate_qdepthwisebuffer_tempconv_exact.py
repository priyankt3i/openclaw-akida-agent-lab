import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_keras import Sequential, layers
from quantizeml.layers import DepthwiseBufferTempConv, InputQuantizer, QuantizedDepthwiseBufferTempConv, QuantizedReLU
from cnn2snn import quantize, convert

SEED = 13
OUT = Path('prototype-1/artifacts/conversion_gate_qdepthwisebuffer_tempconv_exact.json')

np.random.seed(SEED)
tf.random.set_seed(SEED)

report = {'seed': SEED}

float_model = Sequential([
    layers.Input(shape=(1, 1, 4), name='frame'),
    DepthwiseBufferTempConv(kernel_size=3, name='dw_btc'),
    layers.ReLU(max_value=6.0, name='relu6'),
], name='float_dwbtc_relu')
_ = float_model(np.ones((1, 1, 1, 4), dtype=np.float32))
qmodel = quantize(float_model, input_weight_quantization=8, weight_quantization=4, activ_quantization=4)
report['quantize_path'] = {
    'layers': [layer.__class__.__name__ for layer in qmodel.layers],
    'relu_bits': int(qmodel.layers[-1].bitwidth),
}
try:
    convert(qmodel)
    report['quantize_path']['convert'] = 'ok'
except Exception as exc:
    report['quantize_path']['convert'] = {
        'error_type': type(exc).__name__,
        'message': str(exc),
    }

qcfg_dw = {
    'weight_quantizer': {'bitwidth': 4, 'axis': None},
    'output_quantizer': {'bitwidth': 4, 'axis': 'per-tensor', 'signed': True},
    'buffer_bitwidth': 16,
}
qcfg_relu = {
    'output_quantizer': {'bitwidth': 4, 'axis': 'per-tensor', 'signed': False},
}
report['explicit_path'] = {
    'target_layers': ['InputQuantizer', 'QuantizedDepthwiseBufferTempConv', 'QuantizedReLU'],
    'relu_output_bits': 4,
}
try:
    explicit_model = Sequential([
        layers.Input(batch_shape=(1, 1, 1, 1, 4), name='stream'),
        InputQuantizer(bitwidth=8, signed=True, name='input_quantizer'),
        QuantizedDepthwiseBufferTempConv(kernel_size=3, quant_config=qcfg_dw, name='qdw_btc'),
        QuantizedReLU(max_value=6.0, quant_config=qcfg_relu, name='qrelu4'),
    ], name='explicit_qdwbtc_qrelu')
    report['explicit_path']['layers'] = [layer.__class__.__name__ for layer in explicit_model.layers]
    _ = explicit_model(np.ones((1, 1, 1, 1, 4), dtype=np.float32))
    ak_model = convert(explicit_model)
    report['explicit_path']['convert'] = {
        'status': 'ok',
        'akida_model_type': type(ak_model).__name__,
    }
except Exception as exc:
    report['explicit_path']['convert'] = {
        'error_type': type(exc).__name__,
        'message': str(exc),
    }

OUT.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
