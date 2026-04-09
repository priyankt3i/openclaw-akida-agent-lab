import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_keras import Sequential, layers
from quantizeml.layers import InputQuantizer, QuantizedDepthwiseBufferTempConv, QuantizedReLU
from cnn2snn import convert

SEED = 22
KERNEL_SIZE = 2
OUT = Path('prototype-1/artifacts/fusion_probe_qdwbtc_kernel2_4d.json')

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

report = {
    'seed': SEED,
    'kernel_size': KERNEL_SIZE,
    'input_contract': '4D streamed frame input (B,H,W,C)',
}

model = Sequential([
    layers.Input(batch_shape=(1, 1, 1, 4), name='frame'),
    InputQuantizer(bitwidth=8, signed=True, name='input_quantizer'),
    QuantizedDepthwiseBufferTempConv(kernel_size=KERNEL_SIZE, quant_config=qcfg_dw, name='qdw_btc2'),
    QuantizedReLU(max_value=6.0, quant_config=qcfg_relu, name='qrelu6'),
], name='fusion_probe_qdwbtc_kernel2_4d')

report['layers'] = [layer.__class__.__name__ for layer in model.layers]
report['layer_names'] = [layer.name for layer in model.layers]

sample = np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4)
report['sample_shape'] = list(sample.shape)

try:
    y = model(sample, training=False)
    y_values = y.values if hasattr(y, 'values') else y
    report['quantized_call'] = {
        'status': 'ok',
        'output_type': type(y).__name__,
        'output_shape': list(y.shape),
        'values_shape': list(y_values.shape),
    }
except Exception as exc:
    report['quantized_call'] = {
        'status': 'error',
        'error_type': type(exc).__name__,
        'message': str(exc),
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(0)

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
