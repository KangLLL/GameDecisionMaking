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
flags.DEFINE_string('ac_name', 'acn', 'the name used for the actor_critic model directory')

flags.DEFINE_integer('random_seed', 1, 'Random seed to use during training')


flags.DEFINE_integer('update_target_interval', 3000, 'synchronize target network frequency')