import numpy as np
from tensorflow import keras

input_ = np.arange(6)

batch_input = np.reshape(input_, [1, -1, 1])
print(batch_input)
print("----")
model = keras.Sequential([
    keras.layers.Conv1D(1, 3, kernel_initializer=keras.initializers.ones())
])
out = model.predict(x=batch_input)
print(model.summary())
print(out[0])


keras.layers.SeparableConv1D()