import numpy as np
import tensorflow as tf
from tf_keras import Sequential, layers
from quantizeml.layers import DepthwiseBufferTempConv
from cnn2snn import quantize, convert

np.random.seed(0)
tf.random.set_seed(0)

model = Sequential([
    layers.Input(shape=(1, 1, 4)),
    DepthwiseBufferTempConv(kernel_size=3, name='dw_btc'),
    layers.ReLU(max_value=6.0, name='relu6'),
])

_ = model(np.ones((1, 1, 1, 4), dtype=np.float32))
qmodel = quantize(model, input_weight_quantization=8, weight_quantization=4, activ_quantization=4)
print('QMODEL_LAYERS', [layer.__class__.__name__ for layer in qmodel.layers])
convert(qmodel)
print('CONVERT_OK')
