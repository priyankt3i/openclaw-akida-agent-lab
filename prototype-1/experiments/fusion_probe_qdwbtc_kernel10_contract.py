import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_keras import Sequential, layers
from quantizeml.layers import InputQuantizer, QuantizedDepthwiseBufferTempConv, QuantizedReLU
from quantizeml.tensors import FixedPoint
from cnn2snn import convert

SEED = 10
KERNEL_SIZE = 10
ARTIFACT_DIR = Path('prototype-1/artifacts/fusion_probe_qdwbtc_kernel10_contract')
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


def patched_init_fifo(self, new_sample):
    zeros = tf.zeros_like(new_sample)
    expanded = tf.expand_dims(zeros, axis=-2)
    if isinstance(expanded, FixedPoint):
        init_value = FixedPoint(
            tf.tile(expanded.values, [1, 1, 1, 1, self.kernel_size, 1]),
            expanded.value_bits,
            expanded.frac_bits,
        )
    else:
        init_value = tf.tile(expanded, [1, 1, 1, 1, self.kernel_size, 1])
    self._fifo.init_var(init_value)
    if self.counter == 1:
        self._fifo.set_var(init_value)


QuantizedDepthwiseBufferTempConv._init_fifo = patched_init_fifo


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    calibration = np.random.normal(loc=0.0, scale=1.0, size=(8, 1, 1, 1, 4)).astype('float32')
    report = {
        'seed': SEED,
        'kernel_size': KERNEL_SIZE,
        'calibration_shape': list(calibration.shape),
        'patch': 'QuantizedDepthwiseBufferTempConv._init_fifo handles FixedPoint tile via values tensor',
    }

    try:
        model = Sequential([
            layers.Input(batch_shape=(1, 1, 1, 1, 4), name='stream'),
            InputQuantizer(bitwidth=8, signed=True, name='input_quantizer'),
            QuantizedDepthwiseBufferTempConv(kernel_size=KERNEL_SIZE, quant_config=qcfg_dw, name='qdw_btc10'),
            QuantizedReLU(max_value=6.0, quant_config=qcfg_relu, name='qrelu6'),
        ], name='fusion_probe_qdwbtc_kernel10_contract')
        warmup = np.asarray(model(calibration[:1]))
        report['quantize_stage'] = {
            'status': 'ok',
            'warmup_output_shape': list(warmup.shape),
            'layers': [layer.__class__.__name__ for layer in model.layers],
        }
    except Exception as exc:
        report['quantize_stage'] = {
            'status': 'error',
            'error_type': type(exc).__name__,
            'message': str(exc),
        }
        OUT.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    try:
        ak_model = convert(model)
        report['convert_stage'] = {
            'status': 'ok',
            'akida_model_type': type(ak_model).__name__,
            'akida_layers': [layer.__class__.__name__ for layer in ak_model.layers],
            'akida_layer_names': [getattr(layer, 'name', None) for layer in ak_model.layers],
            'akida_model_str': str(ak_model),
        }
    except Exception as exc:
        report['convert_stage'] = {
            'status': 'error',
            'error_type': type(exc).__name__,
            'message': str(exc),
        }

    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
