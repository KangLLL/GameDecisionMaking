# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
sys.path.append("game")
import numpy as np
import wrapped_flappy_bird as env
import random
import time
import cv2

class GameState(object):
    def __init__(self, action_size, rand_seed=0, is_show_score=False, frame_size=84):
        self.rand_seed = rand_seed
        random.seed(self.rand_seed)
        self.action_size = action_size
        self.is_show_score = is_show_score

        self.reset()

        self.steps = 1

        self.reward = 0
        self.terminal = False

        self.frame_size = frame_size

        self.reset()

    def _process_frame(self, action_vector, is_need_reshape):
        reward = 0

        x_t, reward, terminal = self.game.frame_step(action_vector)

        if reward >= 1:
            self.passed_obst += 1
            if self.is_show_score:
                self.full_frame = self.game.full_frame

        x_t = cv2.cvtColor(cv2.resize(x_t, (self.frame_size, self.frame_size)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)

        self.x_t = x_t  # used for visualization

        if is_need_reshape:
            x_t = np.reshape(x_t, (self.frame_size, self.frame_size, 1))

        return x_t, reward, terminal

    def reset(self):
        self.game = env.GameState(self.rand_seed, self.is_show_score)
        self.steps = 1
        self.passed_obst = 0

        x_t, _, _ = self._process_frame(self.vectorize_action(0), False)

        self.reward = 0
        self.terminal = False
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    def vectorize_action(self, action):
        action_vector = np.zeros(self.action_size)
        action_vector[action] = 1
        return action_vector

    def process(self, action):
        action_vector = self.vectorize_action(action)

        x_t1, r, t = self._process_frame(action_vector, True)

        self.reward = r
        self.terminal = t
        self.s_t1 = np.append(self.s_t[:, :, 1:], x_t1, axis=2)

    def update(self):
        self.s_t = self.s_t1
        self.steps += 1

