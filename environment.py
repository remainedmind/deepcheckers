"""
Module to represent game env, that is going to be used for RL
"""
import numpy as np


class CheckersEnvironment:
    def __init__(self, hor, ver):
        self.actions = ["left", "up", "right", "down"]  # 0=Left, 1=Up, 2=right, 3=Down
        self.reward = 0
        self.x = 0
        self.y = 0
        self.done = False
        self.episode_length = 0
        self.no_operation = False
        self.state_observation = [self.x, self.y]

    def reset(self):
        self.done = False
        self.episode_length = 0
        self.x, self.y = 0, 0
        self.state_observation = [self.x, self.y]
        return [self.x, self.y]

    @property
    def action_space(self):
        return self.actions

    def step(self, action):
        if ("game over condition") and action:
            self.done = True
            self.no_operation = False
            return np.array(self.state_observation), self.reward, self.done, self.no_operation, self.episode_length
        elif self.episode_length > 200:
            self.done = True
            self.no_operation = True
            return np.array(self.state_observation), self.reward, self.done, self.no_operation, self.episode_length
        self.action = action
        self.reward = self.get_reward()
        self.state_observation = self.take_action()
        self.episode_length += 1
        self.no_operation = False

    def get_reward(self):
        '''
        Return value : rewards
        Input argument.
        '''
        return 1 if True else -1


    def take_action(self, x):
        return x+1


