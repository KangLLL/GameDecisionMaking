import json
import numpy as np

class Model_free(object):
    SARSA = "sarsa"
    Q_LEARNING = 'q_learning'
    Q_LEARNING_5 = 'q_learning_5'
    Q_LEARNING_10 = 'q_learning_10'
    Q_LEARNING_DOUBLE = 'q_learning_double'
    algo = Q_LEARNING
    addr = 'saved_networks/' + algo

    def __init__(self):
        self.gameCNT = 0 # Game count of current run, incremented after every death
        self.DUMPING_N = 1000 # Number of iterations to dump Q values to JSON after
        self.discount = 1.0
        self.r = {0: 1, 1: -1000} # Reward function
        self.lr = 0.3
        self.load_qvalues()
        if self.algo == self.Q_LEARNING_DOUBLE:
            self.last_state = "420_240_0_0_0"
        else:
            self.last_state = "420_240_0"
        self.last_action = 0
        self.moves = []
        self.epochs = 0
        self.scores = []
        self.times = []

    def get_eopch(self):
        f_score = open(self.addr + '/data/scores.txt', 'r')
        for line in f_score.readlines():
            self.epochs += len(line[1:-2].strip().split(","))
        f_score.close()


    def load_qvalues(self):
        self.qvalues = {}
        try:
            fil = open(self.addr + '/qvalues.json', 'r')
        except IOError:
            return
        self.qvalues = json.load(fil)
        fil.close()

    def act(self, xdif, ydif, vel, xdif2, ydif2):
        state = self.map_state(xdif, ydif, vel, xdif2, ydif2)
        self.moves.append( [self.last_state, self.last_action, state] ) # Add the experience to the history
        self.last_state = state # Update the last_state with the current state

        if not self.qvalues.has_key(state):
            self.qvalues[state] = [0, 0]

        pt = 1 / self.epochs

        if np.random.choice(2, 1, p=[pt, 1 - pt])[0] == 0:
            self.last_action = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        else:
            self.last_action = np.argmax(self.qvalues[state])

        if self.last_action == 0:
            return 0
        else:
            return 1


    def get_last_state(self):
        return self.last_state

    def update_scores(self):
        history = list(reversed(self.moves))

        # Flag if the bird died in the top pipe
        high_death_flag = True if int(history[0][2].split('_')[1]) > 120 else False
        t = 1

        if self.algo == self.SARSA:
            # for exp in history:
            for i in range(len(history)):
                if t == 1:
                    cur_reward = self.r[1]
                else:
                    cur_reward = self.r[0]

                if 1 == len(history):
                    state = history[i][0]
                    action = history[i][1]
                    # Update
                    self.qvalues[state][action] = round((1 - self.lr) * (self.qvalues[state][action]) + \
                                                        self.lr * (cur_reward), 6)
                else:
                    state = history[i][0]
                    action = history[i][1]
                    next_state = history[i - 1][0]
                    next_action = history[i - 1][1]
                    # Update
                    self.qvalues[state][action] = round((1 - self.lr) * (self.qvalues[state][action]) + \
                                                        self.lr * (cur_reward + self.discount * (
                    self.qvalues[next_state][next_action])), 6)

                t += 1
        else:
            # Q-learning score updates
            for exp in history:
                state = exp[0]
                act = exp[1]
                res_state = exp[2]

                # Select reward
                if t == 1 or t == 2:
                    cur_reward = self.r[1]
                elif high_death_flag and act:
                    cur_reward = self.r[1]
                    high_death_flag = False
                else:
                    cur_reward = self.r[0]

                # Update
                self.qvalues[state][act] = round((1-self.lr) * (self.qvalues[state][act]) + \
                                                 self.lr * ( cur_reward + self.discount*max(self.qvalues[res_state]) ), 6)

                t += 1

        self.gameCNT += 1  # increase game count
        self.dump_qvalues()  # Dump q values (if game count % DUMPING_N == 0)
        self.moves = []  # clear history after updating strategies

    def map_state(self, xdif, ydif, vel, xdif2, ydif2):
        if self.algo == self.Q_LEARNING_10:
            xdif = int(xdif) - (int(xdif) % 10)
            ydif = int(ydif) - (int(ydif) % 10)
        elif self.algo == self.Q_LEARNING_5:
            xdif = int(xdif) - (int(xdif) % 5)
            ydif = int(ydif) - (int(ydif) % 5)
        elif self.algo == self.Q_LEARNING_DOUBLE:
            xdif = int(xdif) - (int(xdif) % 10)
            ydif = int(ydif) - (int(ydif) % 10)
            xdif2 = int(xdif2) - (int(xdif2) % 10)
            ydif2 = int(ydif2) - (int(ydif2) % 10)

        if self.algo == self.Q_LEARNING_DOUBLE:
            return str(int(xdif)) + '_' + str(int(ydif)) + '_' + str(vel) + '_' + str(int(xdif2)) + '_' + str(int(ydif2))
        else:
            return str(int(xdif))+'_'+str(int(ydif))+'_'+str(vel)

    def dump_qvalues(self):
        if self.gameCNT % self.DUMPING_N == 0:
            f_qvalue = open(self.addr + '/qvalues.json', 'w')
            json.dump(self.qvalues, f_qvalue)
            f_qvalue.close()

            f_score = open(self.addr + '/data/scores.txt', 'a')
            f_score.writelines(str(self.scores) + '\n')
            f_score.close()

            print('Q-values updated on local file.')
