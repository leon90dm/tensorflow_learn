import keras
import keras.backend as K
import numpy as np

loss_w = K.variable(1)

y_true_w = K.zeros([1, 1])

def my_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred)) * K.cast(loss_w, dtype='float32')


def my_metrice(y_true, y_pred):
    K.set_value(y_true_w, K.eval(y_true))
    return K.mean(K.square(y_true - y_pred))


model = keras.Sequential([
    keras.layers.Dense(1, input_shape=[1])
])
model.compile(optimizer='sgd', loss=my_loss, metrics=[my_metrice])
print("metrices:{}".format(model.metrics_names))

K.set_value(loss_w, 2)
print(model.predict_on_batch(np.ones([1, 1])))
result = model.train_on_batch(np.ones([1, 1]), np.ones([1, 1]))
print(result)
