import keras
import keras.backend as K
import numpy as np


class QLayer(keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(QLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(QLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        V = inputs[0]
        A = inputs[1]
        return V + (A - K.mean(A, axis=1, keepdims=True))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class DuelingDeepQNetwork:
    def __init__(self,
                 memory,
                 n_actions,
                 n_features,
                 using_priority=False,
                 lr=0.005,
                 gamma=0.9,
                 batch_size=10,
                 epsilon=0.01,
                 epsilon_max=0.99,
                 replace_target_periods=200,
                 hidden_size=10):
        self.epsilon = epsilon
        self.epsilon_max = epsilon_max
        self.batch_size = batch_size
        self.gamma = gamma

        self.n_actions = n_actions
        self.n_features = n_features
        self.memory = memory
        self.using_priority = using_priority

        self.hidden_size = hidden_size
        self.learn_step = 0
        self.lr = lr
        self.replace_target_periods = replace_target_periods

    def save_transition(self, s, a, r, flag, s_):
        stack = np.hstack([s, [a, r, flag], s_])
        self.memory.store(stack)

    def build_model(self, trainable):
        kernel_init = keras.initializers.random_normal(0, 0.3)
        bias_init = keras.initializers.constant(0.1)
        f_input = keras.Input(shape=[self.n_features])
        l1 = keras.layers.Dense(self.hidden_size,
                                activation='relu',
                                kernel_initializer=kernel_init,
                                bias_initializer=bias_init,
                                input_shape=[self.n_features],
                                trainable=trainable)(f_input)
        V = keras.layers.Dense(1,
                               kernel_initializer=kernel_init,
                               bias_initializer=bias_init,
                               trainable=trainable)(l1)
        A = keras.layers.Dense(self.n_actions,
                               kernel_initializer=kernel_init,
                               bias_initializer=bias_init,
                               trainable=trainable)(l1)

        # 合并 V 和 A, 为了不让 A 直接学成了 Q, 我们减掉了 A 的均值
        # out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True)) # Q = V(s) + A(s,a)
        Q = QLayer(self.n_actions)([V, A])
        return keras.models.Model(inputs=[f_input],
                                  outputs=[Q])

    def build_net(self):

        self.target_model = self.build_model(False)
        self.eval_model = self.build_model(True)
        rms = keras.optimizers.RMSprop(lr=self.lr)

        loss = 'mse'
        if self.using_priority:
            self.is_weight_input = K.ones(shape=[self.batch_size, 1], dtype='float32')

            def weight_loss(y_true, y_pred):
                return K.mean(self.is_weight_input * K.square(y_true - y_pred))

            loss = weight_loss

        self.eval_model.compile(optimizer=rms, loss=loss)
        print(self.eval_model.summary())

        abs_target_input = keras.Input(shape=[self.n_actions])
        abs_target_output = K.sum(K.abs(abs_target_input - self.eval_model.output), axis=1)
        self.abs_errors = K.function([self.eval_model.input, abs_target_input],
                                     [abs_target_output])  # for updating Sumtree

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            observation = observation[np.newaxis, :]
            pred = self.eval_model.predict_on_batch(observation)
            action = np.argmax(pred)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        if self.learn_step % self.replace_target_periods == 0:
            self.target_model.set_weights(self.eval_model.get_weights())

        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)

        q_next = self.target_model.predict_on_batch(batch_memory[:, -self.n_features:])
        q_eval = self.eval_model.predict_on_batch(batch_memory[:, : self.n_features])

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        flags = batch_memory[:, self.n_features + 2].astype(int)

        q_target[batch_index, eval_act_index] = reward + self.gamma * flags * np.max(q_next, axis=1)

        if self.using_priority:
            K.set_value(self.is_weight_input, ISWeights)

        loss = self.eval_model.train_on_batch(x=batch_memory[:, :self.n_features],
                                              y=q_target)
        if self.using_priority:
            abs_errors = self.abs_errors([batch_memory[:, :self.n_features], q_target])[0]
            self.memory.batch_update(tree_idx, abs_errors)  # update priority

        # increasing epsilon
        self.epsilon = self.epsilon + 0.0005 if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step += 1
        return loss
