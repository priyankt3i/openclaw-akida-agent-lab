import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_keras import Sequential, layers
from quantizeml.layers import DepthwiseBufferTempConv
from cnn2snn import check_model_compatibility, convert, quantize

SEED = 11
BATCH = 2
STEPS = 5
HEIGHT = 1
WIDTH = 1
CHANNELS = 4
KERNEL_SIZE = 3
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT = ARTIFACT_DIR / 'depthwise_buffer_tempconv_probe.json'


def make_stream(batch=BATCH, steps=STEPS, channels=CHANNELS):
    rng = np.random.default_rng(SEED)
    return rng.integers(0, 8, size=(batch, steps, HEIGHT, WIDTH, channels), dtype=np.int32).astype('float32')


def build_model():
    return Sequential([
        layers.Input(shape=(HEIGHT, WIDTH, CHANNELS), name='frame'),
        DepthwiseBufferTempConv(kernel_size=KERNEL_SIZE, name='dw_btc'),
        layers.ReLU(max_value=6.0, name='relu6'),
    ], name='dw_buffer_tempconv_probe')


def stream_model_outputs(model, x):
    outs = []
    for t in range(x.shape[1]):
        outs.append(np.asarray(model(x[:, t], training=False)))
    return np.stack(outs, axis=1)


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    x = make_stream()
    model = build_model()
    float_out = stream_model_outputs(model, x)

    report = {
        'seed': SEED,
        'input_shape': list(x.shape),
        'float_model': {
            'name': model.name,
            'layers': [layer.__class__.__name__ for layer in model.layers],
            'output_shape': list(float_out.shape),
            'output_sum': float(float_out.sum()),
            'output_zero_fraction': float(np.mean(float_out == 0.0)),
        },
        'compatibility': {},
    }

    for input_dtype in ('uint8', 'int8'):
        try:
            check_model_compatibility(model, input_dtype=input_dtype)
            report['compatibility'][input_dtype] = {'status': 'compatible'}
        except Exception as exc:
            report['compatibility'][input_dtype] = {
                'status': 'blocked',
                'error_type': type(exc).__name__,
                'message': str(exc),
            }

    try:
        qmodel = quantize(
            model,
            input_weight_quantization=8,
            weight_quantization=4,
            activ_quantization=4,
        )
        q_out = stream_model_outputs(qmodel, x)
        report['quantized_model'] = {
            'layers': [layer.__class__.__name__ for layer in qmodel.layers],
            'output_shape': list(q_out.shape),
            'output_sum': float(q_out.sum()),
            'output_zero_fraction': float(np.mean(q_out == 0.0)),
        }
        ak_model = convert(qmodel)
        report['conversion'] = {
            'status': 'converted',
            'akida_model': str(ak_model),
            'akida_layers': [layer.__class__.__name__ for layer in ak_model.layers],
        }
    except Exception as exc:
        report['conversion'] = {
            'status': 'blocked',
            'error_type': type(exc).__name__,
            'message': str(exc),
        }

    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
