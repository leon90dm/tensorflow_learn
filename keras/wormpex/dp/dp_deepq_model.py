import keras

from keras.wormpex.dp.config import *
from keras.wormpex.dp.priority_memory import *

# Input
"""
[batch_size, hour of day, 
"""


class DPDeepQNet(object):

    def __init__(self, gamma=0.99):
        self.memory = Memory(MEMORY_CAPACITY)
        self.learn_step_counter = 0
        self.replace_target_iter = 100
        self.gamma = gamma

    def build_model(self):
        self.target = keras.Sequential([
            keras.layers.LSTM(units=LSTM_HIDDEN_SIZE,
                              batch_input_shape=[BATCH_SIZE, TIME_STEPS, FEATURE_INPUT_SIZE],
                              return_sequences=True,
                              trainable=False),
            keras.layers.LSTM(units=LSTM_HIDDEN_SIZE,
                              return_sequences=True,
                              trainable=False),
            keras.layers.TimeDistributed(
                keras.layers.Dense(ACTION_SPACE),
                trainable=False)
        ])

        self.eval = keras.Sequential([
            keras.layers.LSTM(units=LSTM_HIDDEN_SIZE,
                              batch_input_shape=[BATCH_SIZE, TIME_STEPS, FEATURE_INPUT_SIZE],
                              return_sequences=True),
            keras.layers.LSTM(units=LSTM_HIDDEN_SIZE,
                              return_sequences=True),
            keras.layers.TimeDistributed(
                keras.layers.Dense(ACTION_SPACE))
        ])

        rms = keras.optimizers.RMSprop(lr=0.01)
        # [loss, mae]
        self.eval.compile(optimizer=rms, loss="mse",
                          metrics=[keras.metrics.mae])
        print(self.eval.summary())

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        self.memory.store(transition)  # have high priority for newly arrived transition

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target.set_weights(self.eval.get_weights())
            print('\ntarget_params_replaced\n')

        tree_idx, batch_memory, ISWeights = self.memory.sample(BATCH_SIZE)

        q_next = self.target.predict_on_batch(batch_memory[:, -FEATURE_INPUT_SIZE:])
        q_eval = self.eval.predict_on_batch(batch_memory[:, :FEATURE_INPUT_SIZE])

        q_target = q_eval.copy()
        batch_index = np.arange(BATCH_SIZE, dtype=np.int32)
        eval_act_index = batch_memory[:, FEATURE_INPUT_SIZE].astype(int)
        reward = batch_memory[:, FEATURE_INPUT_SIZE + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                                 feed_dict={self.s: batch_memory[:, :self.n_features],
                                                            self.q_target: q_target,
                                                            self.ISWeights: ISWeights})
        self.memory.batch_update(tree_idx, abs_errors)  # update priority

        self.learn_step_counter += 1
