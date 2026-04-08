import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from cnn2snn import quantize
from tf_keras import Sequential, layers

SEED = 23
MODEL_HISTORY = 6
DIM = 16
HIDDEN_1 = 64
HIDDEN_2 = 32
OUT_DIM = 16
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT = ARTIFACT_DIR / 'history_buffer_quant_metadata_probe.json'


def safe_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [safe_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): safe_jsonable(v) for k, v in value.items()}
    if hasattr(value, 'tolist'):
        try:
            return value.tolist()
        except Exception:
            pass
    return repr(value)


def collect_quantizer_like(obj):
    data = {}
    for name in dir(obj):
        if name.startswith('_'):
            continue
        if not any(tok in name.lower() for tok in ['quant', 'scale', 'zero', 'bit', 'sign', 'frac', 'axis', 'range']):
            continue
        try:
            value = getattr(obj, name)
        except Exception as exc:
            data[name] = {'error': repr(exc)}
            continue
        if callable(value):
            continue
        data[name] = safe_jsonable(value)
    if hasattr(obj, 'get_config'):
        try:
            data['get_config'] = safe_jsonable(obj.get_config())
        except Exception as exc:
            data['get_config_error'] = repr(exc)
    if hasattr(obj, 'get_weights'):
        try:
            data['get_weights'] = safe_jsonable(obj.get_weights())
        except Exception as exc:
            data['get_weights_error'] = repr(exc)
    return data


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = Sequential([
        layers.Input(shape=(MODEL_HISTORY * DIM,), name='history_tokens_flat'),
        layers.Dense(HIDDEN_1, use_bias=False, name='proj_in'),
        layers.ReLU(max_value=6.0, name='gate_relu6'),
        layers.Dense(HIDDEN_2, use_bias=False, name='mix_dense'),
        layers.ReLU(max_value=6.0, name='post_relu6'),
        layers.Dense(OUT_DIM, use_bias=False, name='proj_out'),
    ], name='history_buffer_dense_relu6_surrogate')

    qmodel = quantize(model, input_weight_quantization=8, weight_quantization=4, activ_quantization=4)

    report = {
        'model_config': safe_jsonable(qmodel.get_config()),
        'layers': [],
    }

    for layer in qmodel.layers:
        layer_entry = {
            'name': layer.name,
            'type': type(layer).__name__,
            'config': safe_jsonable(layer.get_config()) if hasattr(layer, 'get_config') else None,
            'attrs': collect_quantizer_like(layer),
        }
        for maybe_name in ['quantizer', 'input_quantizer', 'kernel_quantizer', 'bias_quantizer', 'quantized_activation', 'activation']:
            if hasattr(layer, maybe_name):
                try:
                    obj = getattr(layer, maybe_name)
                    layer_entry[maybe_name] = {
                        'type': type(obj).__name__,
                        'details': collect_quantizer_like(obj),
                    }
                except Exception as exc:
                    layer_entry[maybe_name] = {'error': repr(exc)}
        report['layers'].append(layer_entry)

    OUT.write_text(json.dumps(report, indent=2))
    print(str(OUT))


if __name__ == '__main__':
    main()
