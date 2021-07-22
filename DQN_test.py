from maze_env import Maze
from DQN import DQN


def run_maze():
    step = 0
    for episode in range(800):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.chooseAction(observation, 'S', 0.5, 2)
            print("action:", action, "state:", observation )
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.remember_for_rl(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.update_train()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DQN(None, None, 0.5, 'S', 2, env.n_features, env.n_actions)
    env.after(100, run_maze)
    env.mainloop()
