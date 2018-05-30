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
from collections import deque

settings = tf.app.flags.FLAGS


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def createNetwork(network_name, output_dimension):
    with tf.variable_scope(network_name):

        W_conv1 = weight_variable([8, 8, 4, 16])
        b_conv1 = bias_variable([16])

        W_conv2 = weight_variable([4, 4, 16, 32])
        b_conv2 = bias_variable([32])

        W_fc1 = weight_variable([2048, 256])
        b_fc1 = bias_variable([256])

        W_fc2 = weight_variable([256, output_dimension])
        b_fc2 = bias_variable([output_dimension])

        # input layer
        s = tf.placeholder(tf.float32, [None, 80, 80, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

        h_conv3_flat = tf.reshape(h_conv2, [-1, 2048])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1


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


def trainNetwork(s_actor, out_actor, s_critic, out_critic, sess):
    # define the cost function

    sv_p, td, train_critic = trainCritic(out_critic)

    a, td_p, train_actor = trainActor(out_actor)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_" + settings.game + "/readout.txt", 'w')
    h_file = open("logs_" + settings.game + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(settings.action)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # checkpoint = tf.train.get_checkpoint_state("saved_networks")
    # if checkpoint and checkpoint.model_checkpoint_path:
    #     saver.restore(sess, checkpoint.model_checkpoint_path)
    #     print(checkpoint.model_checkpoint_path)
    #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
    # else:
    #     print("Could not find old network weights")

    # start training
    # epsilon = settings.INITIAL_EPSILON
    t = 0

    k = 0
    while True:
        # choose an action with probability

        probs = sess.run(out_actor, {s_actor: [s_t]})  # get probabilities for all actions
        out_a = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int

        a_t = np.zeros([settings.action])
        if t % settings.frame_per_action == 0:
            a_t[out_a] = 1
        else:
            a_t[0] = 1

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > settings.replay_memory:
            D.popleft()

        # only train if done observing
        if t > settings.observe:
        # if t > 100:
            # sample a minibatch to train on
            # if k == 0:
            # minibatch = random.sample(D, settings.batch)
            # k = 1
            minibatch = random.sample(D, settings.batch)
            # minibatch = [D.pop()]

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]

            a_batch = [d[1] for d in minibatch]
            r_batch = np.array([d[2] for d in minibatch])
            # a_batch = np.zeros((len(minibatch),settings.action))
            # a_batch[:,1] = 1
            # r_batch = np.ones(len(minibatch)) * -1

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

        # update the old values
        s_t = s_t1
        t += 1

        print(t)

        # save progress every 10000 iterations
        # if t % 10000 == 0:
        #     saver.save(sess, 'saved_networks/' + config.GAME + '-dqn', global_step=t)

        # print info
        # state = ""
        # if t <= config.OBSERVE:
        #     state = "observe"
        # elif t > config.OBSERVE and t <= config.OBSERVE + config.EXPLORE:
        #     state = "explore"
        # else:
        #     state = "train"
        #
        # print("TIMESTEP", t, "/ STATE", state, \
        #       "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
        #       "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''


def playGame():
    sess = tf.InteractiveSession()

    # build actor network
    s_actor, readout_actor, _ = createNetwork("actor", settings.action)
    with tf.variable_scope("actor"):
        out_actor = tf.clip_by_value(tf.nn.softmax(readout_actor), 1e-20, 1.0)

    # build critic network
    s_critic, readout_critic, _ = createNetwork("critic", 1)

    trainNetwork(s_actor, out_actor, s_critic, readout_critic, sess)


def main():
    playGame()


if __name__ == "__main__":
    main()
