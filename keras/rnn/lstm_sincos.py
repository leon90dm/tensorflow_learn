import keras
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1337)  # for reproducibility

SAMPLE_STEP = 20
BATCH_SIZE = 50
BATCH_START = 0
TIME_STEPS = 20


# def get_batch():
#     global BATCH_START, TIME_STEPS
#     # xs shape (50batch, 20steps)
#     xs = np.linspace(BATCH_START, BATCH_START + SAMPLE_STEP, TIME_STEPS * BATCH_SIZE) \
#         .reshape((BATCH_SIZE, TIME_STEPS))
#     seq = np.sin(xs)
#     res = np.cos(xs)
#     BATCH_START += SAMPLE_STEP
#     return seq, res, xs
def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    return seq, res, xs


def plt_batch(xs, seq, res, pred=None):
    if pred is None:
        plt.plot(xs.flatten(), seq.flatten(), 'k')
    plt.plot(xs.flatten(), res.flatten(), 'b')
    if pred is not None:
        plt.plot(xs.flatten(), pred.flatten(), 'r--')
    plt.show()


model = keras.Sequential([
    keras.layers.LSTM(
        batch_input_shape=[BATCH_SIZE, TIME_STEPS, 1],
        units=20,
        return_sequences=True,
        stateful=True,
        unroll=True
    ),
    keras.layers.TimeDistributed(
        keras.layers.Dense(1)
    )
])

adam = keras.optimizers.Adam(0.006)
model.compile(optimizer=adam,
              loss='mse',
              metrics=['mae'])
print("metrics:{}".format(model.metrics_names))
for i in range(1000):
    x_b, y_b, x = get_batch()
    loss = model.train_on_batch(x=np.expand_dims(x_b, 2),
                                y=np.expand_dims(y_b, 2))
    if i % 100 == 0 and i != 0:
        print("loss at {} : {}".format(i, loss))

seq, res, xs = get_batch()
pred = model.predict_on_batch(x=np.expand_dims(seq, 2))
plt_batch(xs, seq, res, pred)
