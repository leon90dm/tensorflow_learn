import time

from rl.dueling_deep_q.dueling_deep_q_net import DuelingDeepQNetwork
from rl.env.maze_env import Maze

env = Maze()
deep_q = DuelingDeepQNetwork(n_actions=env.n_actions,
                             n_features=env.n_features,
                             batch_size=10,
                             memory_size=2000)
deep_q.build_net()
learning_periods = 10
BATCH_SIZE = 128
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
