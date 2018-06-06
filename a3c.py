# -*- coding: utf-8 -*-
import tensorflow as tf
import multiprocessing
import threading
import numpy as np

import signal
import random
import math
import os
import time

from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier
from statistics import Statistics

import display as DISPLAY
import visualize as VISUALIZE
import config

flags = tf.app.flags

settings = flags.FLAGS

LOG_FILE = 'summaries/{}-{}'.format(settings.experiment_name, settings.agent_type)

random.seed(settings.random_seed)


def log_uniform(lo, hi, size):
    # returns LogUniform(lo,hi) for the number of specified agents.
    return np.logspace(lo, hi, size)


def train_function(parallel_index):
    global global_t

    training_thread = training_threads[parallel_index]
    # set start_time
    start_time = time.time() - wall_t
    training_thread.set_start_time(start_time)

    while True:
        if stop_requested:
            break
        if global_t > settings.max_time_step:
            break

        diff_global_t = training_thread.process(sess, global_t)
        global_t += diff_global_t


def signal_handler(signal, frame):
    global stop_requested
    print('You pressed Ctrl+C!')
    stop_requested = True


def write_checkpoint(saver, start_time):
    global global_t
    global settings

    if not os.path.exists(settings.model_dir):
        os.mkdir(settings.model_dir)
    if not os.path.exists(settings.model_dir + '/' + settings.acn_name):
        os.mkdir(settings.model_dir + '/' + settings.acn_name)

    # write wall time
    wall_t = time.time() - start_time
    wall_t_fname = settings.model_dir + '/' + settings.acn_name + '/' + 'wall_t.' + str(
        global_t)
    with open(wall_t_fname, 'w') as f:
        f.write(str(wall_t))

    saver.save(sess,
               settings.model_dir + '/' + settings.acn_name + '/' + settings.game + '-' + settings.acn_name,
               global_step=global_t)


if __name__ == "__main__":
    device = "/cpu:0"
    if settings.use_gpu:
        device = "/gpu:0"

    num_threads = multiprocessing.cpu_count()
    initial_learning_rates = log_uniform(settings.initial_alpha_low,
                                         settings.initial_alpha_high,
                                         num_threads)

    global_t = 0
    stop_requested = False

    global_network = GameACFFNetwork(settings.action, -1, device)

    learning_rate_input = tf.placeholder("float")
    grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                                  decay=settings.rmsp_alpha,
                                  momentum=0.0,
                                  epsilon=settings.rmsp_epsilon,
                                  clip_norm=settings.grad_norm_clip,
                                  device=device)

    training_threads = []


    for i in range(num_threads):
        training_thread = A3CTrainingThread(i,
                                            global_network,
                                            initial_learning_rates[i],
                                            learning_rate_input,
                                            grad_applier,
                                            settings.max_time_step,
                                            device,
                                            settings.action,
                                            settings.gamma,
                                            settings.local_t_max,
                                            settings.entropy_beta,
                                            settings.agent_type,
                                            settings.performance_log_interval,
                                            settings.log_level,
                                            settings.random_seed)

        training_threads.append(training_thread)

    # prepare session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                            allow_soft_placement=True))

    sess.run(tf.global_variables_initializer())

    # Statistics summary writer
    summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)
    statistics = Statistics(sess, summary_writer, settings.average_summary)

    if settings.agent_type == 'LSTM':
        agent = settings.agent_type
    else:
        agent = 'FF'

    # init or load checkpoint with saver
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(
        settings.model_dir + '/' + settings.acn_name)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded:", checkpoint.model_checkpoint_path)
        tokens = checkpoint.model_checkpoint_path.split("-")
        # set global step
        global_t = int(tokens[2])
        print(">>> global step set: ", global_t)
        # set wall time
        wall_t_fname = settings.model_dir + '/' + settings.acn_name + '/' + 'wall_t.' + str(
            global_t)
        with open(wall_t_fname, 'r') as f:
            wall_t = float(f.read())
        print "Continuing experiment {} with agent type {} at step {}".format(settings.acn_name, agent, global_t)

    else:
        print("Could not find old checkpoint")
        # set wall time
        wall_t = 0.0

        print "Starting experiment {} with agent type {}".format(settings.acn_name, agent)

    train_threads = []
    for i in range(settings.parallel_agent_size):
        train_threads.append(threading.Thread(target=train_function, args=(i,)))

    signal.signal(signal.SIGINT, signal_handler)

    # set start time
    start_time = time.time() - wall_t

    for t in train_threads:
        t.start()

    print('Press Ctrl+C to stop')
    signal.pause()

    print('Now saving data. Please wait')

    for t in train_threads:
        t.join()

    write_checkpoint(saver=saver, start_time=start_time)


