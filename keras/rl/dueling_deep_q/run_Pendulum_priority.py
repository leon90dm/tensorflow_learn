import gym
import numpy as np
from rl.memory.priority_memory import Memory
from rl.dueling_deep_q.dueling_deep_q_net import DuelingDeepQNetwork

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 25
FEATURE_SPACE = 3

memory = Memory(MEMORY_SIZE)

deep_q = DuelingDeepQNetwork(n_actions=ACTION_SPACE,
                             n_features=FEATURE_SPACE,
                             hidden_size=20,
                             batch_size=32,
                             memory=memory,
                             using_priority=True)
deep_q.build_net()

acc_r = 0
total_steps = 0
observation = env.reset()
while total_steps < 95000:
    if deep_q.epsilon >= deep_q.epsilon_max:
        env.render()

    action = deep_q.choose_action(observation)

    f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # [-2 ~ 2] float actions
    observation_, reward, done, info = env.step(np.array([f_action]))

    reward /= 10  # normalize to a range of (-1, 0)
    acc_r = (reward + acc_r)  # accumulated reward

    deep_q.save_transition(observation, action, reward, 0 if done else 1, observation_)

    if total_steps >= MEMORY_SIZE:
        loss = deep_q.learn()
        if total_steps % 100 == 0:
            print("loss & abs_error:{}, epsilon:{}, step:{}, acc_r:{}"
                  .format(loss,
                          deep_q.epsilon,
                          total_steps,
                          acc_r))

    observation = observation_
    total_steps += 1
