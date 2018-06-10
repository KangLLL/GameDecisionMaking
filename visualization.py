import matplotlib.pyplot as plt

Q_LEARNING = 'q_learning'
Q_LEARNING_5 = 'q_learning_5'
Q_LEARNING_10 = 'q_learning_10'
Q_LEARNING_DOUBLE = 'q_learning_double'
NUM_1000 = 1000
NUM_2000 = 2000
NUM_10000 = 10000
BASE_LINE = 150


def read_train_data(algo):
    file_path = 'saved_networks/' + algo + '/data/scores.txt'
    file_score = open(file_path, 'r')
    scores = []
    for line in file_score.readlines():
        score = map(int, line.strip()[1:-1].split(','))
        scores.extend(score)
    file_score.close()

    return scores


def read_evaluation_data(algo):
    if algo == 'dqn':
        file_path = 'saved_networks/dqn.txt'
    else:
        file_path = 'saved_networks/' + algo + '/data/evaluations.txt'
    file_data = open(file_path, 'r')
    score = [0] + map(float, file_data.readline().strip().split(' '))
    file_data.close()

    return score


def scatter_plot(algo):
    train_data = read_train_data(algo)
    eval_data = read_evaluation_data(algo)

    plt.scatter(range(1, len(train_data)+1) ,train_data, marker='.', label='Training')
    plt.plot(range(0, len(train_data)+1, NUM_10000) , eval_data, color='r', linewidth=2.5, label='Evaluation')
    plt.plot([0, len(train_data),], [BASE_LINE, BASE_LINE,], 'k--', color='black', linewidth=2.5, label='Goal 150')

    plt.title('Q Learning')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


def display_plot():
    eval_data_1 = read_evaluation_data(Q_LEARNING)
    plt.plot(range(0, len(eval_data_1)*NUM_10000, NUM_10000) , eval_data_1, color='r', linewidth=2.5, label='1*1')

    eval_data_5 = read_evaluation_data(Q_LEARNING_5)
    plt.plot(range(0, len(eval_data_5)*NUM_1000, NUM_1000) , eval_data_5, color='g', linewidth=2.5, label='5*5')

    eval_data_10 = read_evaluation_data(Q_LEARNING_10)
    plt.plot(range(0, len(eval_data_10)*NUM_1000, NUM_1000) , eval_data_10, color='b', linewidth=2.5, label='10*10')

    eval_data_double = read_evaluation_data(Q_LEARNING_DOUBLE)
    plt.plot(range(0, len(eval_data_double)*NUM_1000, NUM_1000) , eval_data_double, color='c', linewidth=2.5, label='Double')

    plt.plot([0, len(eval_data_double)*NUM_1000, ], [BASE_LINE, BASE_LINE, ], 'k--', color='black', linewidth=2.5, label='Goal 150')

    plt.title('Q Learning')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


def dqn_plot(algo):
    eval_data = read_evaluation_data(algo)

    plt.plot(range(0, len(eval_data)*NUM_2000, NUM_2000) , eval_data, color='r', linewidth=2.5, label='Evaluation')
    plt.plot([0, len(eval_data)*NUM_2000,], [BASE_LINE, BASE_LINE,], 'k--', color='black', linewidth=2.5, label='Goal 150')

    plt.title('DQN')
    plt.xlabel('Update')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # q_learning & sarsa scatter plot
    algo = "q_learning"
    scatter_plot(algo)

    # compare q_learning methods
    display_plot()

    # dqn
    dqn_plot('dqn')
