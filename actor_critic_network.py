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


def create_network():
    # build actor network
    # with tf.variable_scope("actor"):
    #     s_actor, readout_actor = _create_network(settings.action)
    #     out_actor = tf.clip_by_value(tf.nn.softmax(readout_actor), 1e-20, 1.0)
    #
    # # build critic network
    # with tf.variable_scope("critic"):
    #     s_critic, readout_critic = _create_network(1)
    #
    # return s_actor, out_actor, s_critic, readout_critic

    s, h_fc1, _ = fac.build_conv_network()
    W_fc2, b_fc2 = fac.fc_variable([256, settings.action])
    W_fc3, b_fc3 = fac.fc_variable([256, 1])

    o_a = tf.matmul(h_fc1, W_fc2) + b_fc2
    out_actor = tf.clip_by_value(tf.nn.softmax(o_a), 1e-20, 1.0)

    out_critic = tf.matmul(h_fc1, W_fc3) + b_fc3
    return s, out_actor, out_critic


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


def train_network(s, out_actor, out_critic, sess):
    # def train_network(s_actor, out_actor, s_critic, out_critic, sess):
    # define the cost function
    sv_p, a, loss = prepare_loss(out_critic, out_actor)

    train_step = tf.train.AdamOptimizer(5e-7).minimize(loss)
    game_state = GameState(settings.action)

    # saving and loading networks
    saver = tf.train.Saver(max_to_keep=1)
    t = fac.restore_file(sess, saver, settings.acn_name)

    s_batch = []
    a_batch = []
    r_batch = []
    s_prime_batch = []
    v_batch = []

    k_step = 10

    while True:
        # choose an action with probability
        probs = sess.run(out_actor, {s: [game_state.s_t]})[0]  # get probabilities for all actions
        action = np.random.choice(range(len(probs)), p=probs)  # return a int

        # run the selected action and observe next state and reward
        game_state.process(action)

        s_batch.append(game_state.s_t)
        a_batch.append(game_state.vectorize_action(action))
        r_batch.append(game_state.reward)
        s_prime_batch.append(game_state.s_t1)

        if game_state.terminal:
            v_batch.append(game_state.reward)

            for i in range(len(s_batch)):
                I = 1
                v_batch[i] = 0
                for j in range(k_step):
                    if i + j == len(s_batch):
                        break;
                    v_batch[i] += I * r_batch[i + j]
                    I = I * settings.gamma
                if i + k_step < len(s_batch):
                    v_batch[i] += I * v_batch[i + k_step]

            v_batch = np.array(v_batch)

            train_step.run(feed_dict={s: s_batch, sv_p: v_batch.reshape((len(s_batch),1)), a: a_batch})

            s_batch = []
            a_batch = []
            r_batch = []
            s_prime_batch = []
            v_batch = []

        else:
            v_batch.append(settings.gamma * out_critic.eval(feed_dict={s: [game_state.s_t1]}, session=sess)[0][0])

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

        # save progress every 10000 iterations
        # if t % 10000 == 0:
        #     saver.save(sess, settings.model_dir + "/" + settings.acn_name + "/" + settings.game + "-acn", global_step=t)

        # print info
        print("TIMESTEP", t, "/ ACTION", action, "/ REWARD", game_state.reward, \
              "/ ACTION_PROBOBILITY", probs)

        # update the old values
        t += 1
        game_state.update()


def playGame():
    sess = tf.InteractiveSession()
    # s_actor, out_actor, s_critic, out_critic = create_network()
    s, out_actor, out_critic = create_network()
    # train_network(s_actor, out_actor, s_critic, out_critic, sess)
    train_network(s, out_actor, out_critic, sess)


def main():
    playGame()


if __name__ == "__main__":
    main()
