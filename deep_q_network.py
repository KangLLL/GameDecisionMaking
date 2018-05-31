#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys

import random
import numpy as np
import config
import nn_factory as fac
from game_state import GameState
from collections import deque

settings = tf.app.flags.FLAGS

def create_network():
    # network weights
    s, h_fc1 = fac.build_conv_network()
    W_fc2, b_fc2 = fac.fc_variable([256, settings.action])

    # readout layer
    q_out = tf.matmul(h_fc1, W_fc2) + b_fc2
    return s, q_out


def prepare_loss(q_out):
    a = tf.placeholder("float", [None, settings.action])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(q_out, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    return y, a, train_step


def train_network(s, q_out, sess):
    # define the cost function
    y, a, train_step = prepare_loss(q_out)

    # open up a game state to communicate with emulator
    game_state = GameState(settings.action)

    # store the previous observations in replay memory
    D = deque()

    # saving and loading networks
    saver = tf.train.Saver()
    t = fac.restore_file(sess, saver, settings.dqn_name)

    # start training
    epsilon = settings.initial_epsilon
    while True:
        # choose an action epsilon greedily
        readout_t = q_out.eval(feed_dict={s: [game_state.s_t]})[0]

        action = 0
        if t % settings.frame_per_action == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action = random.randrange(settings.action)
            else:
                action = np.argmax(readout_t)

        # scale down epsilon
        if epsilon > settings.final_epsilon and t > settings.observe:
            epsilon -= (settings.initial_epsilon - settings.final_epsilon) / settings.explore

        # run the selected action and observe next state and reward
        game_state.process(action)

        # store the transition in D
        D.append((game_state.s_t, game_state.vectorize_action(action), game_state.reward, game_state.s_t1,
                  game_state.terminal))
        if len(D) > settings.replay_memory:
            D.popleft()

        # only train if done observing
        if t > settings.observe:
            # sample a minibatch to train on
            minibatch = random.sample(D, settings.batch)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = q_out.eval(feed_dict={s: s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + settings.gamma * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch}
            )

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, settings.model_dir + "/" + settings.dqn_name + "/" + settings.game + "-dqn", global_step=t)

        # print info
        state = ""
        if t <= settings.observe:
            state = "observe"
        elif t > settings.observe and t <= settings.observe + settings.explore:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
              "/ EPSILON", epsilon, "/ ACTION", action, "/ REWARD", game_state.reward, \
              "/ Q_MAX %e" % np.max(readout_t))

        # update the old values
        game_state.update()
        t += 1


def playGame():
    sess = tf.InteractiveSession()
    s, q_out = create_network()
    train_network(s, q_out, sess)


def main():
    playGame()


if __name__ == "__main__":
    main()
