"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 10  # grid height
MAZE_W = 10 # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        self.canvas.create_rectangle(
            origin[0] - 20, origin[1] - 20 + UNIT,
            origin[0] + 20 + UNIT, origin[1] +20 + UNIT,
            fill='black')
        self.canvas.create_rectangle(
            origin[0] - 20 + UNIT*3, origin[1] - 20 + UNIT,
            origin[0] + 20+ UNIT*3, origin[1] +20 + UNIT*3,
            fill='black')
        self.canvas.create_rectangle(
            origin[0] - 20 + UNIT, origin[1] - 20 + UNIT*3,
            origin[0] + 20+ UNIT*2, origin[1] +20 + UNIT*3,
            fill='black')
        self.canvas.create_rectangle(
            origin[0] - 20 + UNIT*5, origin[1] - 20,
            origin[0] + 20+ UNIT*5, origin[1] +20 + UNIT*2,
            fill='black')
        self.canvas.create_rectangle(
            origin[0] - 20 + UNIT*6, origin[1] - 20 + UNIT*2,
            origin[0] + 20+ UNIT*7, origin[1] +20 + UNIT*2,
            fill='black')
        self.canvas.create_rectangle(
            origin[0] - 20 + UNIT*9, origin[1] - 20,
            origin[0] + 20+ UNIT*9, origin[1] +20 + UNIT,
            fill='black')
        self.canvas.create_rectangle(
            origin[0] - 20 + UNIT*8, origin[1] - 20 + UNIT*4,
            origin[0] + 20+ UNIT*9, origin[1] +20 + UNIT*4,
            fill='black')
        self.canvas.create_rectangle(
            origin[0] - 20, origin[1] - 20 + UNIT*5,
            origin[0] + 20, origin[1] +20 + UNIT*5,
            fill='black')
        self.canvas.create_rectangle(
            origin[0] - 20 + UNIT*2, origin[1] - 20 + UNIT*5,
            origin[0] + 20 + UNIT*2, origin[1] + 20 + UNIT*7,
            fill='black')
        self.canvas.create_rectangle(
            origin[0] - 20, origin[1] - 20 + UNIT*7,
            origin[0] + 20 + UNIT, origin[1] + 20 + UNIT*7,
            fill='black')
        self.canvas.create_rectangle(
            origin[0] - 20 + UNIT*4, origin[1] - 20 + UNIT*5,
            origin[0] + 20 + UNIT*6, origin[1] + 20 + UNIT*5,
            fill='black')
        self.canvas.create_rectangle(
            origin[0] - 20 + UNIT*6, origin[1] - 20 + UNIT*6,
            origin[0] + 20 + UNIT*6, origin[1] + 20 + UNIT*7,
            fill='black')
        self.canvas.create_rectangle(
            origin[0] - 20 + UNIT*8, origin[1] - 20 + UNIT*6,
            origin[0] + 20 + UNIT*8, origin[1] + 20 + UNIT*6,
            fill='black')
        self.canvas.create_rectangle(
            origin[0] - 20 + UNIT*5, origin[1] - 20 + UNIT*9,
            origin[0] + 20 + UNIT*5, origin[1] + 20 + UNIT*9,
            fill='black')
        self.canvas.create_rectangle(
            origin[0] - 20 + UNIT*8, origin[1] - 20 + UNIT*8,
            origin[0] + 20 + UNIT*8, origin[1] + 20 + UNIT*8,
            fill='black')
        

        # create oval
        oval_center = origin + UNIT * 9
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        s_ = [s[0]+base_action[0], s[1]+base_action[1], s[2]+base_action[0], s[3]+base_action[1]]
        #self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        #s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 100
            done = True
            s_ = 'terminal'
        elif [(s_[0]+s_[2])/2, (s_[1]+s_[3])/2] in [[20,60],[60,60],[140,60],[140,100],[140,140],
        [60,140],[100,140],[220,20],[220,60],[220,100],[260,100],[300,100],[380,20],[380,60],
        [340,180],[380,180],[20,220],[100,220],[100,260],[100,300],[60,300],[20,300],[180,220],
        [220,220],[260,220],[260,260],[260,300],[340,260],[340,340],[220,380]]:
        	reward = -5
        	done = False
        	s_ = s
        	base_action = np.array([0, 0])
#        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
#            reward = -1
 #           done = True
 #           s_ = 'terminal'
        else:
            reward = -1
            done = False

        self.canvas.move(self.rect, base_action[0], base_action[1])

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()