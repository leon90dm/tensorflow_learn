import keras
import keras.backend as K
from rl.env.maze_env import Maze
from rl.memory.priority_memory import *
import time


class PriorityDeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 memory_size,
                 lr=0.01,
                 gamma=0.9,
                 batch_size=10,
                 epsilon=0.01,
                 epsilon_max=0.99,
                 hidden_size=10):
        self.epsilon = epsilon
        self.epsilon_max = epsilon_max
        self.batch_size = batch_size
        self.gamma = gamma

        self.n_actions = n_actions
        self.n_features = n_features
        self.memory = Memory(memory_size)

        self.hidden_size = hidden_size
        self.learn_step = 0
        self.lr = lr

    def save_transition(self, s, a, r, flag, s_):
        stack = np.hstack([s, [a, r, flag], s_])
        self.memory.store(stack)

    def build_net(self):

        self.target_model = keras.Sequential([
            keras.layers.Dense(self.hidden_size, input_shape=[self.n_features],
                               trainable=False),
            keras.layers.Dense(self.n_actions, trainable=False)
        ])
        self.eval_model = keras.Sequential([
            keras.layers.Dense(self.hidden_size, input_shape=[self.n_features]),
            keras.layers.Dense(self.n_actions)
        ])
        rms = keras.optimizers.RMSprop(lr=self.lr)

        self.is_weight_var = K.variable(np.ones(
            (self.batch_size, 1)))

        def weighted_mse(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred), axis=-1)

        self.eval_model.compile(optimizer=rms, loss=weighted_mse)
        print(self.eval_model.summary())

        abs_target_input = keras.Input(shape=[self.n_actions])
        abs_target_output = K.sum(K.abs(abs_target_input - self.eval_model.output), axis=1)
        self.abs_errors = K.function([self.eval_model.input, abs_target_input],
                                     [abs_target_output])  # for updating Sumtree

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            observation = observation[np.newaxis, :]
            pred = self.eval_model.predict_on_batch(observation)[0]
            action = np.argmax(pred)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        if self.learn_step % 5 == 0:
            self.target_model.set_weights(self.eval_model.get_weights())
            # increasing epsilon
            self.epsilon = self.epsilon + 0.1 if self.epsilon < self.epsilon_max else self.epsilon_max

        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)

        q_next = self.target_model.predict_on_batch(batch_memory[:, -self.n_features:])
        q_eval = self.eval_model.predict_on_batch(batch_memory[:, : self.n_features])

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        flags = batch_memory[:, self.n_features + 2].astype(int)

        q_target[batch_index, eval_act_index] = reward + self.gamma * flags * np.max(q_next, axis=1)

        K.set_value(self.is_weight_var, ISWeights)
        loss = self.eval_model.train_on_batch(x=batch_memory[:, :self.n_features],
                                              y=q_target)
        abs_errors = self.abs_errors([batch_memory[:, :self.n_features], q_target])[0]

        self.memory.batch_update(tree_idx, abs_errors)  # update priority

        self.learn_step += 1
        return loss, np.mean(ISWeights)


BATCH_SIZE = 128
env = Maze()
deep_q = PriorityDeepQNetwork(n_actions=env.n_actions,
                              n_features=env.n_features,
                              batch_size=BATCH_SIZE,
                              memory_size=2000)
deep_q.build_net()
learning_periods = 10
max_steps = BATCH_SIZE


def run_maze():
    g_step = 0
    for episode in range(10000):
        # initial observation
        observation = env.reset()
        step = 0
        while step < max_steps:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = deep_q.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            if deep_q.epsilon >= deep_q.epsilon_max:
                if g_step % learning_periods == 0:
                    print("eval hidden:{}".format(deep_q.eval_model.predict_on_batch(observation[np.newaxis, :])[0]))
                    time.sleep(0.1)
            else:
                deep_q.save_transition(observation, action, reward, 0 if done else 1, observation_)

            if g_step > BATCH_SIZE and g_step % learning_periods == 0:
                loss = deep_q.learn()
                print("loss & abs_error:{}, epsilon:{}"
                      .format(loss,
                              deep_q.epsilon))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            g_step += 1
            step += 1

    # end of game
    print('game over')
    env.destroy()


env.after(1000, run_maze)
env.mainloop()
