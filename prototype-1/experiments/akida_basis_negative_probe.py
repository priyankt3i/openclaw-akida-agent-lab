import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from cnn2snn import quantize, convert
from tf_keras import Sequential, layers

SEED = 23
DIM_IN = 96
ARTIFACT_DIR = Path('prototype-1/artifacts')
OUT = ARTIFACT_DIR / 'akida_basis_negative_probe.json'


def build_model():
    return Sequential([
        layers.Input(shape=(DIM_IN,), name='history_tokens_flat'),
        layers.Dense(64, use_bias=False, name='proj_in'),
        layers.ReLU(max_value=6.0, name='gate_relu6'),
        layers.Dense(32, use_bias=False, name='mix_dense'),
        layers.ReLU(max_value=6.0, name='post_relu6'),
        layers.Dense(16, use_bias=False, name='proj_out'),
    ], name='history_buffer_dense_relu6_surrogate')


def rel_mse(ref, cand):
    return float(np.mean((cand - ref) ** 2) / (np.mean(ref ** 2) + 1e-12))


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()
    qmodel = quantize(model, input_weight_quantization=8, weight_quantization=4, activ_quantization=4)
    ak_model = convert(qmodel)

    float_vals = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    q_outs = {}
    for val in float_vals:
        x = np.zeros((1, DIM_IN), dtype=np.float32)
        x[0, 0] = val
        q_outs[str(val)] = np.asarray(qmodel(x, training=False)).reshape(-1)

    ak_outs = {}
    for u in range(16):
        x = np.zeros((1, DIM_IN), dtype=np.uint8)
        x[0, 0] = u
        ak_outs[str(u)] = np.asarray(ak_model.predict(x)).reshape(-1)

    matches = {}
    for fkey, fout in q_outs.items():
        ranked = []
        for ukey, uout in ak_outs.items():
            ranked.append({
                'uint_value': int(ukey),
                'rel_mse': rel_mse(fout, uout),
                'mean_output': float(np.mean(uout)),
                'nonzero_fraction': float(np.mean(uout != 0.0)),
            })
        ranked.sort(key=lambda row: row['rel_mse'])
        matches[fkey] = ranked[:6]

    # Summarize whether negatives collapse to the same Akida bucket.
    negative_best = {
        fkey: matches[fkey][0]['uint_value']
        for fkey in ['-8.0', '-4.0', '-2.0', '-1.0', '-0.5']
    }
    positive_best = {
        fkey: matches[fkey][0]['uint_value']
        for fkey in ['0.0', '0.5', '1.0', '2.0', '4.0', '8.0']
    }

    report = {
        'proj_in_weight_range': {
            'min': int(np.asarray(ak_model.layers[1].variables['weights']).min()),
            'max': int(np.asarray(ak_model.layers[1].variables['weights']).max()),
        },
        'negative_best_uint_matches': negative_best,
        'positive_best_uint_matches': positive_best,
        'match_table': matches,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
