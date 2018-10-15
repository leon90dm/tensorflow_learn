import keras
import numpy as np

from rl.env.maze_env import Maze


class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 memory_size,
                 gamma=0.9,
                 batch_size=30,
                 epsilon=0.01,
                 epsilon_max=0.99,
                 hidden_size=10):
        self.epsilon = epsilon
        self.epsilon_max = epsilon_max
        self.batch_size = batch_size
        self.gamma = gamma

        self.n_actions = n_actions
        self.n_features = n_features
        self.memory = np.zeros([memory_size, n_features * 2 + 3])
        self.memory_size = memory_size
        self.memory_offset = 0
        self.hidden_size = hidden_size
        self.learn_step = 0

    def save_transition(self, s, a, r, flag, s_):
        stack = np.hstack([s, [a, r, flag], s_])
        self.memory[self.memory_offset % self.memory_size, :] = stack
        self.memory_offset += 1

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
        rms = keras.optimizers.RMSprop(lr=0.01)
        loss = keras.losses.mse
        self.eval_model.compile(optimizer=rms, loss=loss)
        print(self.eval_model.summary())

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            observation = observation[np.newaxis, :]
            pred = self.eval_model.predict_on_batch(observation)[0]
            action = np.argmax(pred)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        if self.learn_step % 100 == 0:
            self.target_model.set_weights(self.eval_model.get_weights())
            print('\ntarget_params_replaced\n')
        if self.memory_offset < self.memory_size:
            sample_index = np.random.choice(self.memory_offset, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        q_next = self.target_model.predict_on_batch(batch_memory[:, -self.n_features:])
        q_eval = self.eval_model.predict_on_batch(batch_memory[:, : self.n_features])

        q_target = q_eval.copy()

        reward = batch_memory[:, self.n_features + 1]
        flags = batch_memory[:, self.n_features + 2]
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        q_target[np.arange(len(batch_memory)), eval_act_index] = reward + self.gamma * np.max(q_next, axis=1) * flags
        loss = self.eval_model.train_on_batch(batch_memory[:, : self.n_features], y=q_target)

        # increasing epsilon
        self.epsilon = self.epsilon + 0.005 if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step += 1
        return loss


env = Maze()
deep_q = DeepQNetwork(n_actions=env.n_actions,
                      n_features=env.n_features,
                      batch_size=200,
                      memory_size=2000)
deep_q.build_net()


def run_maze():
    step = 0
    for episode in range(1000):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = deep_q.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            deep_q.save_transition(observation, action, reward, 0 if done else 1, observation_)

            if step > 200 and step % 10 == 0:
                if deep_q.epsilon < deep_q.epsilon_max:
                    loss = deep_q.learn()
                    if step % 50 == 0:
                        print("loss:{}, epsilon:{}, mem_offset:{}"
                              .format(loss,
                                      deep_q.epsilon,
                                      deep_q.memory_offset))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


env.after(1000, run_maze)
env.mainloop()
