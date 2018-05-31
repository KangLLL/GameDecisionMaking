#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys

sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
import config
import nn_factory as fac
from game_state import GameState
from collections import deque

settings = tf.app.flags.FLAGS

def _create_network(network_name, output_dimension):
    with tf.variable_scope(network_name):
        s, h_fc1 = fac.build_conv_network()
        W_fc2, b_fc2 = fac.fc_variable([256, output_dimension])

        out = tf.matmul(h_fc1, W_fc2) + b_fc2
        return s, out

def create_network():
    # build actor network
    s_actor, readout_actor = _create_network("actor", settings.action)
    with tf.variable_scope("actor"):
        out_actor = tf.clip_by_value(tf.nn.softmax(readout_actor), 1e-20, 1.0)

    # build critic network
    s_critic, readout_critic = _create_network("critic", 1)
    return s_actor, out_actor, s_critic, readout_critic

def trainCritic(out_critic):
    state_value = tf.placeholder(tf.float32, [None, 1])

    td_error = state_value - out_critic
    loss = tf.square(td_error)
    train_step = tf.train.AdamOptimizer(5e-7).minimize(loss)

    return state_value, td_error, train_step


def trainActor(out_actor):
    a = tf.placeholder(tf.float32, [None, settings.action])
    td_error = tf.placeholder(tf.float32, [None], "td")

    log_prob = tf.log(tf.reduce_sum(tf.multiply(out_actor, a), reduction_indices=1))
    loss = tf.reduce_sum(tf.multiply(td_error, log_prob))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(-loss)

    return a, td_error, train_step


def train_network(s_actor, out_actor, s_critic, out_critic, sess):
    # define the cost function
    sv_p, td, train_critic = trainCritic(out_critic)
    a, td_p, train_actor = trainActor(out_actor)

    game_state = GameState(settings.action)

    # store the previous observations in replay memory
    D = deque()

    # saving and loading networks
    saver = tf.train.Saver(max_to_keep=None)
    t = fac.restore_file(sess, saver, settings.ac_name)

    while True:
        # choose an action with probability
        probs = sess.run(out_actor, {s_actor: [game_state.s_t]})[0]  # get probabilities for all actions
        action = np.random.choice(range(len(probs)), p=probs)  # return a int

        # run the selected action and observe next state and reward
        game_state.process(action)

        # store the transition in D
        D.append((game_state.s_t, game_state.vectorize_action(action), game_state.reward, game_state.s_t1,
                  game_state.terminal))
        if len(D) > settings.replay_memory:
            D.popleft()

        # only train if done observing
        if t > settings.observe and len(D) > settings.batch:
            minibatch = random.sample(D, settings.batch)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = np.array([d[2] for d in minibatch])
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = out_critic.eval(feed_dict={s_critic: s_j1_batch}, session=sess)
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + settings.gamma * readout_j1_batch[i, 0])
            y_batch = np.array(y_batch)

            # print("before:")
            # print(out_critic.eval(feed_dict={s_critic: s_j_batch}, session=sess))
            # print(out_actor.eval(feed_dict={s_actor: s_j_batch}, session=sess))
            td_batch = td.eval(feed_dict={s_critic: s_j_batch, sv_p: y_batch.reshape((len(minibatch), 1))}, session=sess)

            # perform gradient step
            train_critic.run(feed_dict={s_critic: s_j_batch, sv_p: y_batch.reshape((len(minibatch), 1))})
            # actor_train_step.run(feed_dict={s_actor: s_j_batch})
            train_actor.run(feed_dict={s_actor: s_j_batch, a: a_batch, td_p: td_batch.reshape(-1)})

            # print("after:")
            # print("value:")
            # print(out_critic.eval(feed_dict={s_critic: s_j_batch}, session=sess))
            # print(out_critic.eval(feed_dict={s_critic: s_j1_batch}, session=sess))
            # print("r:")
            # print(r_batch)
            # print("actor:")
            # print(out_actor.eval(feed_dict={s_actor: s_j_batch}, session=sess))

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, settings.model_dir + "/" + settings.ac_name + "/" + settings.game + "-acn", global_step=t)

        # print info
        print("TIMESTEP", t, "/ ACTION", action, "/ REWARD", game_state.reward, \
              "/ ACTION_PROBOBILITY", probs)

        # update the old values
        t += 1
        game_state.update()


def playGame():
    sess = tf.InteractiveSession()
    s_actor, out_actor, s_critic, out_critic = create_network()
    train_network(s_actor, out_actor, s_critic, out_critic, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
