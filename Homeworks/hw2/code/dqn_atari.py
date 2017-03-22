#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import keras.layers.convolutional as C
import keras.layers.core as core
from keras.layers import Input, Dense
from keras.models import Model

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.constants import *
from deeprl_hw2.preprocessors import *
from deeprl_hw2.utils import *
from deeprl_hw2.policy import *
from deeprl_hw2.core import *
import gym


def create_model(window, input_shape, num_actions,
                 model_name):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    if model_name == 0:
        return linear_model(window, input_shape, num_actions)
    elif model_name == 1:
        return linear_double_model(window, input_shape, num_actions)
    elif model_name == 2:
        return deep_model(window, input_shape, num_actions)
    elif model_name == 3:
        return double_deep(window, input_shape, num_actions)
    elif model_name == 4:
        return dueling_deep(window, input_shape, num_actions)
    else:
        print "No suitable models found."


def linear_model(window, input_shape, num_actions):
    input = Input(shape=(input_shape) + (window,), name='input')
    flattened_input = Flatten()(input)
    with tf.name_scope('output'):
        output = Dense(num_actions, activation='linear')(flattened_input)
    model = Model(inputs=input, outputs=output, name='linear_q_network')
    print model.summary()
    return model


def deep_model(window, input_shape, num_actions):
    print (input_shape) + (window,)
    inputs = Input(shape=(input_shape) + (window,))
    first_layer = C.Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation='relu')(inputs)
    second_layer = C.Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation='relu')(first_layer)
    output_layer = core.Dense(units=num_actions)(second_layer)
    model = Model(inputs=inputs, outputs=output_layer)
    return model


def linear_double_model():
    pass

def double_deep():
    pass

def dueling_deep():
    pass


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Game')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name', required=True)
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    print 'Using Tensorflow Version of ' + tf.__version__
    #args.input_shape = tuple(args.input_shape)

    args.output = get_output_folder(args.output, args.env)
    print "Output Folder: " + args.output

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

    sess = tf.Session()
    K.set_session(sess)

    env = gym.make(args.env)
    #env = gym.wrappers.Monitor(env, args.output + '/gym')
    num_actions = env.action_space.n
    # 0 linear; 1 linear double; 2 deep; 3 double deep; 4 dueling deep
    model = create_model(WINDOW, INPUT_SHAPE, num_actions, 2)
    atari_preprocessor = AtariPreprocessor(INPUT_SHAPE)
    history_preprocessor = HistoryPreprocessor(4)
    preprocessor = PreprocessorSequence([atari_preprocessor, history_preprocessor])
    memory = ReplayMemory(MAX_MEMORY, WINDOW)
    policy = GreedyEpsilonPolicy(0.05)

    dqn_agent = DQNAgent(model, num_actions, preprocessor, memory, policy, GAMMA,
                         TARGET_UPDATE_FREQ, INIT_MEMORY, TRAIN_FREQ, BATCH_SIZE)

    optimizer = Adam(lr=0.00025, epsilon=10-3)
    loss_func = mean_huber_loss
    dqn_agent.compile(optimizer, loss_func)
    #dqn_agent.calc_q_values(state)
    dqn_agent.fit(env, 100000,  MAX_EPISODE_LENGTH)


if __name__ == '__main__':
    main()
