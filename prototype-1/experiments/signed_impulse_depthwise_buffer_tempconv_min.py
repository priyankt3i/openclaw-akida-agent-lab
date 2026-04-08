import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_keras import Sequential, layers
from quantizeml.layers import DepthwiseBufferTempConv, reset_buffers
from quantizeml.models.quantize import QuantizationParams, quantize
from cnn2snn import convert

SEED = 0
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT = ARTIFACT_DIR / 'signed_impulse_depthwise_buffer_tempconv_min.json'


def run_tf_stream(model, seq):
    reset_buffers(model)
    outputs = []
    for value in seq:
        x = np.array([[[[value]]]], dtype=np.float32)
        y = np.asarray(model(x, training=False)).reshape(-1)[0]
        outputs.append(float(y))
    return outputs


def run_akida_stream(model, seq):
    outputs = []
    for value in seq:
        x = np.array([[[[value]]]], dtype=np.int8)
        y = np.asarray(model.predict(x)).reshape(-1)[0]
        outputs.append(float(y))
    return outputs


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = Sequential([
        layers.Input(batch_shape=(1, 1, 1, 1), name='frame'),
        DepthwiseBufferTempConv(kernel_size=3, name='dw_btc'),
    ])
    _ = model(np.zeros((1, 1, 1, 1), dtype=np.float32))
    model.layers[0].set_weights([np.array([[1.0], [1.0], [1.0]], dtype=np.float32)])

    seq = np.array([-5.0, 0.0, 0.0, 0.0], dtype=np.float32)
    float_out = run_tf_stream(model, seq)

    samples = np.array([-5.0, 0.0, 5.0, 0.0, 1.0, -1.0], dtype=np.float32).reshape(6, 1, 1, 1)
    qparams = QuantizationParams(input_dtype='int8', activation_bits=4, weight_bits=4, input_weight_bits=8)
    qmodel = quantize(model, qparams=qparams, samples=samples, batch_size=1)
    quant_out = run_tf_stream(qmodel, seq)

    ak_model = convert(qmodel)
    akida_out = run_akida_stream(ak_model, seq.astype(np.int8))

    report = {
        'sequence': seq.tolist(),
        'kernel': [1.0, 1.0, 1.0],
        'quantized_layers': [layer.__class__.__name__ for layer in qmodel.layers],
        'float_output': float_out,
        'quantized_output': quant_out,
        'akida_output': akida_out,
        'all_akida_outputs_negative_during_impulse_tail': bool(all(v < 0 for v in akida_out[:3])),
        'akida_clears_to_zero_after_fifo_flush': bool(akida_out[3] == 0.0),
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
