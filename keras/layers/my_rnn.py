import keras
import keras.backend as K
from keras.layers import RNN
import numpy as np


class MyRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MyRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=[input_shape[-1], self.units],
                                      initializer='uniform')
        self.rnn_kernel = self.add_weight('rnn_kernel',
                                          shape=[self.units, self.units],
                                          initializer='uniform')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.rnn_kernel)
        return output, [output]


# Let's use this cell in a RNN layer:

m = keras.Sequential([
    RNN(MyRNNCell(5))
])

input_ = np.arange(10)
y = m.predict(x=input_.reshape([1, 5, 2]))
print(y)

# Here's how to use the cell to build a stacked RNN:

# cells = [MyRNNCell(32), MyRNNCell(64)]
# x = keras.Input((None, 5))
# layer = RNN(cells)
# y = layer(x)
