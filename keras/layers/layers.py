import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

plt.scatter()

input_ = np.arange(12)

batch_input = input_[np.newaxis, :, np.newaxis]

dens_layer = keras.layers.Dense(1,
                                input_shape=(len(input_), 1),
                                use_bias=False,
                                kernel_initializer=keras.initializers.ones)

# layer.set_weights(layer.get_weights())

model = keras.Sequential([dens_layer])
out = model.predict(x=batch_input)
# print(out[0])


model1 = keras.Sequential([
    keras.layers.Dropout(0.999)
])
# print(model1.predict(batch_input))

batch_input = np.reshape(input_, [2, 2, 3])
print("before fill: %s" % batch_input)
batch_input[:, 1, :] = 0
print("before mask: %s" % batch_input)
mask_model = keras.Sequential([
    keras.layers.Masking(mask_value=0, input_shape=(2, 3))
])
print("after mask")
print(mask_model.predict(batch_input))
