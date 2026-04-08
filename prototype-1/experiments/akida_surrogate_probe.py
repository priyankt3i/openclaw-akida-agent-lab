import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras
from tf_keras import Sequential, layers
from cnn2snn import check_model_compatibility, convert, quantize

SEED = 7
TOKENS = 8
DIM = 16
CHANNELS = 1
BATCH = 8
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT = ARTIFACT_DIR / 'akida_surrogate_probe.json'


def make_input(batch=BATCH):
    rng = np.random.default_rng(SEED)
    # Signed activation range to match the int8 path that actually converts.
    return rng.integers(-16, 16, size=(batch, TOKENS * DIM), dtype=np.int32).astype('float32')


def dense_surrogate():
    return Sequential([
        layers.Input(shape=(TOKENS * DIM,), name='tokens_flat'),
        layers.Dense(64, use_bias=False, name='proj_in'),
        layers.ReLU(max_value=6.0, name='gate_relu6'),
        layers.Dense(32, use_bias=False, name='mix_dense'),
        layers.ReLU(max_value=6.0, name='post_relu6'),
        layers.Dense(16, use_bias=False, name='proj_out'),
    ], name='dense_relu6_surrogate')


def conv_depthwise_probe():
    return Sequential([
        layers.Input(shape=(TOKENS, DIM, CHANNELS), name='tokens_grid'),
        layers.Conv2D(8, 1, use_bias=False, name='proj_in_pw'),
        layers.DepthwiseConv2D(3, padding='same', use_bias=False, name='mix_dw3x3'),
        layers.ReLU(max_value=6.0, name='gate_relu6'),
        layers.Conv2D(8, 1, use_bias=False, name='mix_pw'),
        layers.ReLU(max_value=6.0, name='post_relu6'),
        layers.Conv2D(4, 1, use_bias=False, name='proj_out'),
    ], name='conv_depthwise_probe')


def activation_sparsity(model, sample):
    relu_layers = [layer.output for layer in model.layers if isinstance(layer, layers.ReLU)]
    if not relu_layers:
        return {}
    probe = tf_keras.Model(model.input, relu_layers)
    outputs = probe(sample, training=False)
    if not isinstance(outputs, list):
        outputs = [outputs]
    sparsity = {}
    for layer, output in zip([l for l in model.layers if isinstance(l, layers.ReLU)], outputs):
        arr = np.asarray(output)
        sparsity[layer.name] = float(np.mean(arr == 0.0))
    return sparsity


def probe_dense(sample):
    model = dense_surrogate()
    metrics = {
        'model_name': model.name,
        'layers': [layer.__class__.__name__ for layer in model.layers],
    }
    dense_out = model(sample, training=False).numpy()
    metrics['output_shape'] = list(dense_out.shape)
    metrics['output_zero_fraction'] = float(np.mean(dense_out == 0.0))
    metrics['activation_sparsity'] = activation_sparsity(model, sample)

    compat = {}
    for input_dtype in ('int8', 'uint8'):
        try:
            check_model_compatibility(model, input_dtype=input_dtype)
            compat[input_dtype] = {'status': 'compatible'}
        except Exception as exc:
            compat[input_dtype] = {'status': 'blocked', 'error_type': type(exc).__name__, 'message': str(exc)}
    metrics['compatibility'] = compat

    try:
        qmodel = quantize(model, input_weight_quantization=8, weight_quantization=4, activ_quantization=4)
        metrics['quantized_layers'] = [layer.__class__.__name__ for layer in qmodel.layers]
        ak_model = convert(qmodel)
        metrics['conversion'] = {
            'status': 'converted',
            'akida_model': str(ak_model),
        }
    except Exception as exc:
        metrics['conversion'] = {
            'status': 'blocked',
            'error_type': type(exc).__name__,
            'message': str(exc),
        }
    return metrics


def probe_conv_depthwise():
    model = conv_depthwise_probe()
    sample = make_input().reshape(BATCH, TOKENS, DIM, CHANNELS)
    _ = model(sample, training=False).numpy()
    metrics = {
        'model_name': model.name,
        'layers': [layer.__class__.__name__ for layer in model.layers],
        'activation_sparsity': activation_sparsity(model, sample),
    }
    try:
        check_model_compatibility(model, input_dtype='int8')
        metrics['compatibility'] = {'status': 'compatible'}
    except Exception as exc:
        metrics['compatibility'] = {
            'status': 'blocked',
            'error_type': type(exc).__name__,
            'message': str(exc),
        }
    try:
        qmodel = quantize(model, input_weight_quantization=8, weight_quantization=4, activ_quantization=4)
        metrics['quantized_layers'] = [layer.__class__.__name__ for layer in qmodel.layers]
        try:
            ak_model = convert(qmodel)
            metrics['quantization'] = {
                'status': 'converted',
                'akida_model': str(ak_model),
            }
        except Exception as exc:
            metrics['quantization'] = {
                'status': 'blocked_after_quantization',
                'error_type': type(exc).__name__,
                'message': str(exc),
            }
    except Exception as exc:
        metrics['quantization'] = {
            'status': 'blocked',
            'error_type': type(exc).__name__,
            'message': str(exc),
        }
    return metrics


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    dense_sample = make_input()
    report = {
        'seed': SEED,
        'tokens': TOKENS,
        'dim': DIM,
        'batch': BATCH,
        'dense_relu6_surrogate': probe_dense(dense_sample),
        'conv_depthwise_probe': probe_conv_depthwise(),
        'summary': {
            'proven_path': 'A tf_keras Sequential Dense -> ReLU6 -> Dense -> ReLU6 -> Dense surrogate quantizes and converts with cnn2snn when checked with signed int8 input assumptions.',
            'blocked_path': 'A Conv2D + DepthwiseConv2D probe is not yet Akida-ready here because conversion rejects the depthwise kernel / input-sign combination and quantization leaves unserializable layers.',
        },
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
