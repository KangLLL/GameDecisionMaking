import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('game', 'bird', 'the name used for log files')
flags.DEFINE_integer('action', 2, 'number of valid actions')
flags.DEFINE_float('gamma', 0.99, 'decay rate of past observations')
flags.DEFINE_integer('observe', 1000, 'number of iterations before training')
flags.DEFINE_integer('explore', 2000000, 'frames over which to anneal epsilon')
flags.DEFINE_float('final_epsilon', 0.0001, 'final value of epsilon')
flags.DEFINE_float('initial_epsilon', 0.0001, 'starting value of epsilon')
flags.DEFINE_integer('replay_memory', 50000, 'number of transitions to remember')
flags.DEFINE_integer('batch', 32, 'size of minibatch')
flags.DEFINE_integer('frame_per_action', 1, 'the minimum interval frames between two actions')

flags.DEFINE_integer('evaluate_episodes', 10000, 'number of iterations for each checkpoint')

flags.DEFINE_string('model_dir', 'saved_networks', 'the name used for the parent directory')
flags.DEFINE_string('dqn_name', 'dqn', 'the name used for the dqn model directory')
flags.DEFINE_string('acn_name', 'acn', 'the name used for the actor_critic model directory')

flags.DEFINE_integer('random_seed', 1, 'Random seed to use during training')


flags.DEFINE_integer('update_target_interval', 3000, 'synchronize target network frequency')


# EXPERIMENT
flags.DEFINE_string('experiment_name', 'flappybird', 'Name of the current experiment (for summary)')
flags.DEFINE_boolean('use_gpu', False, 'If GPU should be used to speed up the training process')

# AGENT
flags.DEFINE_integer('parallel_agent_size', 16, 'Number of parallel agents')
flags.DEFINE_string('agent_type', 'FF', 'What type of A3C to train the agent with [FF, LSTM] (default FF)')

# TRAINING
flags.DEFINE_integer('max_time_step', 40000000, 'Maximum training steps')
flags.DEFINE_float('initial_alpha_low', -5, 'LogUniform low limit for learning rate (represents x in 10^x)')
flags.DEFINE_float('initial_alpha_high', -3, 'LogUniform high limit for learning rate (represents x in 10^x)')
flags.DEFINE_float('entropy_beta', 0.01, 'Entropy regularization constant')
flags.DEFINE_float('grad_norm_clip', 40.0, 'Gradient norm clipping')

# OPTIMIZER
flags.DEFINE_float('rmsp_alpha', 0.99, 'Decay parameter for RMSProp')
flags.DEFINE_float('rmsp_epsilon', 0.1, 'Epsilon parameter for RMSProp')
flags.DEFINE_integer('local_t_max', 256, 'Repeat step size')

# LOG
flags.DEFINE_string('log_level', 'FULL', 'Log level [NONE, FULL]')
flags.DEFINE_integer('average_summary', 25, 'How many episodes to average summary over')
flags.DEFINE_integer('performance_log_interval', 1000, 'How often to print current performance (in steps/s)')

# DISPLAY
flags.DEFINE_integer('display_episodes', 50, 'Numbers of episodes to display')
flags.DEFINE_integer('display_time_sleep', 0, 'Sleep time in each state (seconds)')
flags.DEFINE_string('display_log_level', 'MID',
                    'Display log level - NONE prints end summary, MID prints episode summary and FULL prints for every state [NONE, MID, FULL]')
flags.DEFINE_boolean('display_save_log', False, 'If MID level log should be saved')
flags.DEFINE_boolean('show_max', True, 'If a screenshot of the high score should be plotted')