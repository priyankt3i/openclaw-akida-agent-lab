import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_keras import Model, layers
from quantizeml.layers import DepthwiseBufferTempConv, reset_buffers
from quantizeml.models.quantize import QuantizationParams, quantize
from cnn2snn import convert

SEED = 11
EXTERNAL_WIDTH = 16
BOTTLENECK_WIDTH = 8
KERNEL_SIZE = 4
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT = ARTIFACT_DIR / 'kernel4_qdwbtc_double_stack_probe.json'


def build_model():
    x_in = layers.Input(batch_shape=(1, 1, 1, EXTERNAL_WIDTH), name='frame')
    x = layers.Conv2D(BOTTLENECK_WIDTH, kernel_size=1, use_bias=False, name='proj_in_1')(x_in)
    x = DepthwiseBufferTempConv(kernel_size=KERNEL_SIZE, name='dw_btc4_1')(x)
    x = layers.Conv2D(EXTERNAL_WIDTH, kernel_size=1, use_bias=False, name='proj_out_1')(x)
    x = layers.Conv2D(BOTTLENECK_WIDTH, kernel_size=1, use_bias=False, name='proj_in_2')(x)
    x = DepthwiseBufferTempConv(kernel_size=KERNEL_SIZE, name='dw_btc4_2')(x)
    x = layers.Conv2D(EXTERNAL_WIDTH, kernel_size=1, use_bias=False, name='proj_out_2')(x)
    return Model(x_in, x, name='kernel4_qdwbtc_double_stack_probe')


def init_weights(model):
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            kernel = np.zeros(layer.kernel.shape, dtype=np.float32)
            out_ch = layer.filters
            in_ch = layer.kernel.shape[2]
            for idx in range(min(in_ch, out_ch)):
                kernel[0, 0, idx, idx] = 1.0
            layer.set_weights([kernel])
        elif isinstance(layer, DepthwiseBufferTempConv):
            kernel = np.ones((KERNEL_SIZE, layer.weights[0].shape[1]), dtype=np.float32)
            layer.set_weights([kernel])


def run_tf_stream(model, seq):
    reset_buffers(model)
    outputs = []
    for frame in seq:
        x = frame.reshape(1, 1, 1, -1).astype(np.float32)
        y = np.asarray(model(x, training=False)).reshape(-1)
        outputs.append(y.tolist())
    return outputs


def run_akida_stream(model, seq):
    outputs = []
    for frame in seq:
        x = frame.reshape(1, 1, 1, -1).astype(np.int8)
        y = np.asarray(model.predict(x)).reshape(-1)
        outputs.append(y.tolist())
    return outputs


def summarize_temporal_layers(layer_names, layer_classes):
    hits = []
    for name, cls in zip(layer_names, layer_classes):
        label = f'{name}:{cls}'
        low = label.lower()
        if 'temp' in low or 'buffer' in low or 'btc' in low:
            hits.append(label)
    return hits


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()
    _ = model(np.zeros((1, 1, 1, EXTERNAL_WIDTH), dtype=np.float32))
    init_weights(model)

    seq = np.zeros((6, EXTERNAL_WIDTH), dtype=np.float32)
    seq[0, :BOTTLENECK_WIDTH] = 1.0
    seq[1, :BOTTLENECK_WIDTH] = -1.0
    seq[2, BOTTLENECK_WIDTH:] = 2.0

    float_out = run_tf_stream(model, seq)

    calibration = np.random.randint(-3, 4, size=(32, 1, 1, EXTERNAL_WIDTH)).astype(np.float32)
    qparams = QuantizationParams(input_dtype='int8', activation_bits=4, weight_bits=4, input_weight_bits=8)
    qmodel = quantize(model, qparams=qparams, samples=calibration, batch_size=1)
    quant_out = run_tf_stream(qmodel, seq)

    ak_model = convert(qmodel)
    akida_out = run_akida_stream(ak_model, seq.astype(np.int8))

    q_layers = [layer.__class__.__name__ for layer in qmodel.layers]
    q_names = [layer.name for layer in qmodel.layers]
    ak_layers = [layer.__class__.__name__ for layer in ak_model.layers]
    ak_names = [getattr(layer, 'name', '') for layer in ak_model.layers]
    ak_layer_reprs = [repr(layer) for layer in ak_model.layers]
    temporal_hits = summarize_temporal_layers(ak_names, ak_layers)

    report = {
        'seed': SEED,
        'shape_contract': 'Streamed 4D samples [B,H,W,C] with internal FIFO in each QuantizedDepthwiseBufferTempConv.',
        'architecture': {
            'external_width': EXTERNAL_WIDTH,
            'bottleneck_width': BOTTLENECK_WIDTH,
            'kernel_size': KERNEL_SIZE,
            'blocks': 2,
            'pattern': ['Conv2D(1x1,16->8)', 'DepthwiseBufferTempConv(k=4)', 'Conv2D(1x1,8->16)'] * 2,
        },
        'sequence_shape': list(seq.shape),
        'quantized_layers': q_layers,
        'quantized_layer_names': q_names,
        'akida_layers': ak_layers,
        'akida_layer_names': ak_names,
        'akida_temporal_hits': temporal_hits,
        'akida_layer_reprs': ak_layer_reprs,
        'float_output_last_frame': float_out[-1],
        'quantized_output_last_frame': quant_out[-1],
        'akida_output_last_frame': akida_out[-1],
        'float_output_sum_per_frame': [float(np.sum(v)) for v in float_out],
        'quantized_output_sum_per_frame': [float(np.sum(v)) for v in quant_out],
        'akida_output_sum_per_frame': [float(np.sum(v)) for v in akida_out],
        'akida_model_summary': str(ak_model),
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
