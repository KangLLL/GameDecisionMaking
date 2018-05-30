import tensorflow as tf
import numpy as np
import cv2
import deep_q_network as dqn
import config
import wrapped_flappy_bird as game

from tensorflow.python.tools import inspect_checkpoint as chkp
import config

settings = tf.app.flags.FLAGS

def evaluateNetwork(s, readout, h_fc1, sess, evaluate_model_path):
    # define the cost function
    a = tf.placeholder("float", [None, settings.action])

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # printing
    # a_file = open("logs_" + config.GAME + "/readout.txt", 'w')
    # h_file = open("logs_" + config.GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(settings.action)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    if evaluate_model_path:
        saver.restore(sess, evaluate_model_path)
        # chkp.print_tensors_in_checkpoint_file(evaluate_model_path, tensor_name="", all_tensors=True)
        print("Successfully loaded:", evaluate_model_path)
    else:
        sess.run(tf.global_variables_initializer())
        print("Could not find evaluate network weights")

    # start evaluating
    episode = 0
    t = 0
    while episode < settings.evaluate_iterations:
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([settings.action])
        action_index = 0

        if t % settings.frame_per_action == 0:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t = np.append(x_t1, s_t[:, :, :3], axis=2)

        # update the old values
        if terminal:
            t = 0
            episode += 1
        else:
            t += 1

def evaluate():
    session = tf.InteractiveSession()
    s, readout, h_fc1 = dqn.createNetwork()
    evaluateNetwork(s, readout, h_fc1, session, "saved_networks/bird-dqn-130000")

    # evaluateNetwork(s, readout, h_fc1, session, "")

if __name__ == "__main__":
    evaluate()
    print("evaluate")