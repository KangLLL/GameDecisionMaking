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


def _create_network(output_dimension):
    s, h_fc1, _ = fac.build_conv_network()
    W_fc2, b_fc2 = fac.fc_variable([256, output_dimension])

    out = tf.matmul(h_fc1, W_fc2) + b_fc2
    return s, out

def create_two_network():
    # build actor network
    with tf.variable_scope("actor"):
        s_actor, readout_actor = _create_network(settings.action)
        out_actor = tf.clip_by_value(tf.nn.softmax(readout_actor), 1e-20, 1.0)

    # build critic network
    with tf.variable_scope("critic"):
        s_critic, readout_critic = _create_network(1)

    return s_actor, out_actor, s_critic, readout_critic

def create_shared_network():
    s, h_fc1, _ = fac.build_conv_network()
    W_fc2, b_fc2 = fac.fc_variable([256, settings.action])
    W_fc3, b_fc3 = fac.fc_variable([256, 1])

    o_a = tf.matmul(h_fc1, W_fc2) + b_fc2
    out_actor = tf.clip_by_value(tf.nn.softmax(o_a), 1e-20, 1.0)

    out_critic = tf.matmul(h_fc1, W_fc3) + b_fc3
    return s, out_actor, out_critic


def prepare_critic_loss(out_critic):
    R = tf.placeholder(tf.float32, [None, 1])
    td = R - out_critic

    loss = tf.nn.l2_loss(td)
    return R, td, loss

def prepare_actor_loss(out_actor):
    td = tf.placeholder(tf.float32, [None, 1])
    a = tf.placeholder(tf.float32, [None, settings.action])
    log_pi = tf.log(out_actor)
    loss = -tf.reduce_sum(tf.multiply(log_pi, a), reduction_indices=1) * td

    return td, a, loss

def prepare_loss(out_critic, out_actor):
    state_value = tf.placeholder(tf.float32, [None, 1])
    td_error = state_value - out_critic

    value_loss = 0.5 * tf.nn.l2_loss(td_error)

    a = tf.placeholder(tf.float32, [None, settings.action])

    log_pi = tf.log(tf.clip_by_value(out_actor, 1e-20, 1.0))
    entropy = - tf.reduce_sum(out_actor * log_pi, reduction_indices=1)

    policy_loss = - tf.reduce_sum(
        tf.reduce_sum(tf.multiply(log_pi, a), reduction_indices=1) * td_error + entropy * 0.1)

    total_loss = value_loss + policy_loss

    return state_value, a, total_loss


def train_network(s_actor, out_actor, s_critic, out_critic, sess):
    # def train_network(s_actor, out_actor, s_critic, out_critic, sess):
    # define the cost function

    R_p, td_v, loss_c = prepare_critic_loss(out_critic)
    td_p, a_p, loss_a = prepare_actor_loss(out_actor)


    train_actor = tf.train.AdamOptimizer(1e-6).minimize(loss_a)
    train_critic = tf.train.AdamOptimizer(2e-6).minimize(loss_c)

    # sv_p, a, loss = prepare_loss(out_critic, out_actor)
    #
    # train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)
    game_state = GameState(settings.action)

    # saving and loading networks
    saver = tf.train.Saver(max_to_keep=None)
    t = fac.restore_file(sess, saver, settings.acn_name)
    # t = 0

    s_batch = []
    a_batch = []
    r_batch = []
    s_prime_batch = []
    v_batch = []
    R_batch = []

    # k_step = 1

    while True:
        # choose an action with probability
        probs = sess.run(out_actor, {s_actor: [game_state.s_t]})[0]  # get probabilities for all actions
        action = np.random.choice(range(len(probs)), p=probs)  # return a int

        # run the selected action and observe next state and reward
        game_state.process(action)

        s_batch.append(game_state.s_t)
        a_batch.append(game_state.vectorize_action(action))
        r_batch.append(game_state.reward)
        s_prime_batch.append(game_state.s_t1)
        v_batch.append(sess.run(out_critic, {s_critic:[game_state.s_t]})[0][0])

        if game_state.terminal:
            s_batch.reverse()
            a_batch.reverse()
            r_batch.reverse()
            s_prime_batch.reverse()
            v_batch.reverse()

            R = 0
            for i in range(len(s_batch)):
                R = r_batch[i] + R * settings.gamma
                R_batch.append([R])

            td_batch = td_v.eval(feed_dict={s_critic: s_batch, R_p: R_batch })
            train_critic.run(feed_dict={s_critic: s_batch, R_p: R_batch})
            train_actor.run(feed_dict={s_actor: s_batch, td_p: td_batch, a_p: a_batch})

            # train_step.run(feed_dict={s: s_batch, sv_p: v_batch.reshape((len(s_batch),1)), a: a_batch})

            s_batch = []
            a_batch = []
            r_batch = []
            s_prime_batch = []
            v_batch = []
            R_batch = []

        # else:
        #     v_batch.append(settings.gamma * out_critic.eval(feed_dict={s: [game_state.s_t1]}, session=sess)[0][0])

            # y = game_state.reward + settings.gamma * out_critic.eval(feed_dict={s: [game_state.s_t1]}, session=sess)[0]

        # train_step.run(feed_dict={s:[game_state.s_t], sv_p:[[y]], a:[game_state.vectorize_action(action)]})

        # train_step.run(feed_dict={s: [game_state.s_t], sv_p: [y], a: [game_state.vectorize_action(action)]})

        # print("after:")
        # print("value:")
        # print(out_critic.eval(feed_dict={s_critic: s_j_batch}, session=sess))
        # print(out_critic.eval(feed_dict={s_critic: s_j1_batch}, session=sess))
        # print("r:")
        # print(r_batch)
        # print("actor:")
        # print(out_actor.eval(feed_dict={s_actor: s_j_batch}, session=sess))

        # print info
        print("TIMESTEP", t, "/ ACTION", action, "/ REWARD", game_state.reward, \
              "/ ACTION_PROBOBILITY", probs)

        # update the old values
        t += 1
        game_state.update()

        # save progress every 10000 iterations
        if t % 20000 == 0:
            saver.save(sess, settings.model_dir + "/" + settings.acn_name + "/" + settings.game + "-acn", global_step=t)


def playGame():
    sess = tf.InteractiveSession()
    s_actor, out_actor, s_critic, out_critic = create_two_network()
    # s, out_actor, out_critic = create_network()
    # train_network(s_actor, out_actor, s_critic, out_critic, sess)
    train_network(s_actor, out_actor, s_critic, out_critic, sess)


def main():
    playGame()


if __name__ == "__main__":
    main()
