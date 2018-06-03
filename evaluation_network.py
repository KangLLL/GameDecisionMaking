import tensorflow as tf
import numpy as np
import config
import random
import time
import matplotlib.pyplot as plt
import sys
import deep_q_network as dqn
# import actor_critic_network as acn

from game_ac_network import GameACFFNetwork


np.set_printoptions(threshold='nan')
from game_state import GameState

settings = tf.app.flags.FLAGS

def initialize_network(sess, method, file_name):
    if method == 0:
        with tf.variable_scope('predict'):
            s_pre, q_out_pre, vs_pre = dqn.create_network()
        # with tf.variable_scope('target'):
        #     s_tar, q_out_tar, vs_tar = dqn.create_network()
        # with tf.variable_scope('predict'):
        #     y, a, train_step = dqn.prepare_loss(q_out_pre)
        result = (s_pre, q_out_pre)
    elif method == 1:
        network = GameACFFNetwork(settings.action, 1, device="/cpu:0")
        result = network

    saver = tf.train.Saver()

    model_file_name = settings.model_dir + '/' + method_2_name(method) + '/' + file_name
    saver.restore(sess, model_file_name)
    print('Successfully loaded:', model_file_name)

    return result


def choose_action(method, sess, agent, s_values):
    if method == 0:
        q_values = sess.run(agent[1], {agent[0]: [s_values]})[0]
        return np.argmax(q_values)
    elif method == 1:
        pi_, value_ = agent.run_policy_and_value(sess, s_values)
        return np.random.choice(range(len(pi_)), p=pi_)
    return 0


def display(t, method, rand_seed, agent):
    log_file_path = 'log_{}.txt'.format(t)
    log_file_path = settings.model_dir + '/' + method_2_name(method) + '/' + log_file_path

    episode = 0
    terminal = False

    episode_rewards = []
    episode_steps = []
    episode_passed_obsts = []
    print ' '
    print 'DISPLAYING {} EPISODES'.format(settings.evaluate_episodes)
    print '--------------------------------------------------- '

    while not episode == settings.evaluate_episodes:
        episode_reward = 0
        episode_passed_obst = 0

        game_state = GameState(settings.action, rand_seed, is_show_score=True)
        print 'EPISODE {}'.format(episode)

        full_frame = None
        while True:
            action = choose_action(method, sess, agent, game_state.s_t)
            game_state.process(action)
            terminal = game_state.terminal
            episode_step = game_state.steps
            reward = game_state.reward
            passed_obst = game_state.passed_obst
            if len(episode_passed_obsts) == 0:
                if passed_obst > 0:
                    full_frame = game_state.full_frame
            elif episode_passed_obst > np.max(episode_passed_obsts):
                full_frame = game_state.full_frame

            episode_reward += reward
            episode_passed_obst = passed_obst

            if not terminal:
                game_state.update()
            else:
                break

        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        episode_passed_obsts.append(episode_passed_obst)

        reward_steps = format(float(episode_reward) / float(episode_step), '.4f')
        print "EPISODE: {}  /  STEPS: {}  /  PASSED OBST: {}  /  REWARD: {}  /  REWARD/STEP: {}".format(episode,
                                                                                                        episode_step,
                                                                                                        passed_obst,
                                                                                                        episode_reward,
                                                                                                        reward_steps)

        with open(log_file_path, "a") as text_file:
            text_file.write(
                '{},{},{},{},{}\n'.format(episode, episode_step, passed_obst, episode_reward, reward_steps))

        episode += 1

    print '--------------------------------------------------- '
    print 'DISPLAY SESSION FINISHED'
    print 'TOTAL EPISODES: {}'.format(settings.evaluate_episodes)
    print ' '
    print 'MIN'
    print 'REWARD: {}  /  STEPS: {}  /  PASSED OBST: {}'.format(np.min(episode_rewards), np.min(episode_steps),
                                                                np.min(episode_passed_obsts))
    print ' '
    print 'AVERAGE'
    print  'REWARD: {}  /  STEPS: {}  /  PASSED OBST: {}'.format(np.average(episode_rewards), np.average(episode_steps),
                                                                 np.average(episode_passed_obsts))
    print ' '
    print 'MAX'
    print 'REWARD: {}  /   STEPS: {}  /   PASSED OBST: {}'.format(np.max(episode_rewards), np.max(episode_steps),
                                                                  np.max(episode_passed_obsts))

    print ' '
    print 'AVERAGE_REWARD'
    print 'REWARD: {}  /  PASSED OBST: {}'.format(np.sum(episode_rewards) / settings.evaluate_episodes,
                                                  np.sum(episode_passed_obsts) / settings.evaluate_episodes)

    with open(log_file_path, "a") as text_file:
        text_file.write(
            '{},{}\n'.format(np.sum(episode_rewards) / settings.evaluate_episodes,
                             np.sum(episode_passed_obsts) / settings.evaluate_episodes))

def method_2_name(method):
    return settings.dqn_name if method == 0 else settings.acn_name

if __name__ == '__main__':
    method = 1  # 0: dpn 1: ac
    t = 3510781
    if len(sys.argv) > 1:
        method = int(sys.argv[1])
    if len(sys.argv) > 2:
        t = int(sys.argv[2])
    file_name = settings.game + '-' + method_2_name(method) + '-' + str(t)

    sess = tf.Session()
    agent = initialize_network(sess, method, file_name)

    display(t, method, 1, agent)
