"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QVLearningTable:
    def __init__(self, actions, learning_rate=0.1, lr_v=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.lr_v = lr_v
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)#pd.read_pickle('./models/model.pkl')#
        self.v_table = pd.DataFrame(columns=[0], dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.v_table.loc[s_,0]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
            
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        #update V function
        self.v_table.loc[s,0] = self.v_table.loc[s,0] + self.lr_v * (r + self.gamma*self.v_table.loc[s_,0] - self.v_table.loc[s,0])
        if s_ == 'terminal':
            self.q_table.to_pickle('./models/model_qv-Q.pkl')
            self.q_table.to_pickle('./models/model_qv-V.pkl')

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
            self.v_table = self.v_table.append(
                pd.Series(
                    [0],
                    index=self.v_table.columns,
                    name=state,
                )
            )
           