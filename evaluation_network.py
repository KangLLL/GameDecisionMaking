import tensorflow as tf
import numpy as np
import config
import random
import time
import matplotlib.pyplot as plt
import sys
import deep_q_network as dqn
import actor_critic_network as acn

np.set_printoptions(threshold='nan')
from game_state import GameState

settings = tf.app.flags.FLAGS

def initialize_network(sess, method, file_name):
    if method == 0:
        s, out = dqn.create_network()
        method_name = settings.dpn_name
        result = (s, out)
    elif method == 1:
        s_a, o_a, s_c, o_c = acn.create_network()
        method_name = settings.acn_name
        result = (s_a, o_a, s_c, o_c)

    saver = tf.train.Saver()

    model_file_name = settings.model_dir + '/' + method_name + '/' + file_name
    saver.restore(sess, model_file_name)
    print('Successfully loaded:', model_file_name)

    return result


def choose_action(method, sess, s, out, s_values):
    if method == 0:
        q_values = sess.run(out, {s: [s_values]})[0]
        return np.argmax(q_values)
    elif method == 1:
        probs = sess.run(out, {s: [s_values]})[0]
        return np.random.choice(range(len(probs)), p=probs)
    return 0


def display(t, method, rand_seed, s, o):
    log_file_path = 'log_{}.txt'.format(t)
    if method == 0:
        log_file_path = settings.model_dir + '/' + settings.dpn_name + '/' + log_file_path
    elif method == 1:
        log_file_path = settings.model_dir + '/' + settings.acn_name + '/' + log_file_path

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

        game_state = GameState(rand_seed, settings.action, show_score=True)
        print 'EPISODE {}'.format(episode)

        full_frame = None
        while True:
            action = choose_action(method, sess, s, o, game_state.s_t)
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


if __name__ == '__main__':
    method = 0  # 0: dpn 1: ac
    t = 10000
    if len(sys.argv) > 1:
        method = int(sys.argv[1])
    if len(sys.argv) > 2:
        t = int(sys.argv[2])
    file_name = settings.game + '-' + method + '-' + str(t)

    sess = tf.Session()
    if method == 0:
        s, o = initialize_network(sess, method, file_name)
    else:
        s, o, _, _ = initialize_network(sess, method, file_name)

    display(t, method, 1, s, o)
