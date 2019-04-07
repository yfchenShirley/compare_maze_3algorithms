"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       walls       [reward = -5].
Yellow bin circle:      exit    [reward = +100].
All other states:       ground      [reward = -1].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from Qlearning_brain import QLearningTable
from Sarsa_brain import SarsaTable


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

EPIS = 200

def update_Q_learning():
    global plot_y
    plot_y.clear()

    for episode in range(EPIS):       
        # initial observation
        observation = env.reset()
        reward_total = 0
        

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            reward_total += reward

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                plot_y.append(reward_total)
                print(f"Episode {episode}:")
                print(f"Total rewards(Q-learning): {reward_total}")
                break

    # end of game
    print('game over for Q-learning')
    env.destroy()


def update_Sarsa():
    global plot_y
    plot_y.clear()

    for episode in range(EPIS):
        
        # initial observation
        observation = env.reset()
        reward_total = 0

        # RL choose action based on observation
        action = RL.choose_action(str(observation))
        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            reward_total += reward

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                plot_y.append(reward_total)
                print(f"Episode {episode}:")
                print(f"Total rewards(Sarsa): {reward_total}")
                break

    # end of game
    
    print('game over for Sarsa')
    env.destroy()



if __name__ == "__main__":
    plot_y = list(range(EPIS))
    fig, ax = plt.subplots()
    # Data for plotting
    ax.set(xlabel='episode (s)', ylabel='Total Rewards',
           title='Total rewards at each episode')
    ax.grid()

    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)), learning_rate=0.5, e_greedy=0.85)
    env.after(100, update_Q_learning)
    env.mainloop()
    ax.plot(range(EPIS), plot_y, label='Q-learning')

    #Sarsa
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)), learning_rate=0.1,  e_greedy=0.9)
    env.after(100, update_Sarsa)
    env.mainloop()
    ax.plot(range(EPIS), plot_y, label='Sarsa')


    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')    
    fig.savefig("Qlearning_vs_Sarsa.png")
    plt.show()