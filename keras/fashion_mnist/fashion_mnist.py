import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from matplotlib.font_manager import FontProperties

font = FontProperties(fname='/Library/Fonts/Songti.ttc', size=10)

# https://www.tensorflow.org/tutorials/keras/basic_classification

# ~/.keras/datasets/fashion-mnist
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T恤', '裤子', '套衫', '连衣裙', '外套',
               '凉鞋', '衬衫', '运动鞋', '包包', '短靴']

train_images = train_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])


def plot_value_array(img, predictions_array):
    argmax = int(np.argmax(predictions_array))
    cn = class_names[argmax]
    print("predict:{}".format(cn))
    f = plt.subplot(111)
    f.set_title(cn, fontproperties=font)
    plt.imshow(img)
    plt.show()


CKPT = "fashion_mnist.cktp"
if os.path.isfile(CKPT):
    model.load_weights("fashion_mnist.cktp")
tb_callback = keras.callbacks.TensorBoard(log_dir="./graphs", histogram_freq=0, write_graph=True, write_images=True)

while True:
    cmd = input("command:")
    if cmd == "fit":
        model.fit(train_images, train_labels, epochs=1, callbacks=[tb_callback])
    elif cmd == "eval":
        test_images_ = test_images / 255.0
        test_loss, test_acc = model.evaluate(test_images_, test_labels)
        print('Test accuracy:', test_acc)
    elif cmd == "quit":
        break
    else:
        choice = np.random.choice(len(test_images))
        img = test_images[choice]
        tlabel = test_labels[choice]
        predictions_single = model.predict(np.expand_dims(img / 255.0, 0))
        plot_value_array(img, predictions_single[0])

model.save("fashion_mnist.cktp")
