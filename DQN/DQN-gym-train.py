#!/usr/bin/env python
# -*- coding: utf-8 -*-
# E-DRL
# Author: Music

from tensorpack.tfutils import symbolic_functions as symbf

import argparse
from tensorpack.predict.common import PredictConfig
from tensorpack import *
from tensorpack.models.model_desc import ModelDesc, InputVar
from tensorpack.train.config import TrainConfig
from tensorpack.tfutils.common import *
from tensorpack.callbacks.group import Callbacks
from tensorpack.callbacks.stat import StatPrinter
from tensorpack.callbacks.common import ModelSaver
from tensorpack.callbacks.param import ScheduledHyperParamSetter, HumanHyperParamSetter
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.RL.expreplay import ExpReplay
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.queue import QueueInputTrainer
from tensorpack.RL.common import MapPlayerState, show_images
from tensorpack.RL.gymenv import GymEnv
from tensorpack.RL.common import LimitLengthPlayer, PreventStuckPlayer
from tensorpack.RL.history import HistoryFramePlayer
from tensorpack.tfutils.argscope import argscope
from tensorpack.models.conv2d import Conv2D
from tensorpack.models.pool import MaxPooling
from tensorpack.models.nonlin import LeakyReLU, PReLU
from tensorpack.models.fc import FullyConnected
import tensorpack.tfutils.summary as summary
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.callbacks.graph import RunOp
from tensorpack.callbacks.base import PeriodicCallback

import common
from common import play_model, Evaluator, eval_model_multithread
import numpy as np

BATCH_SIZE = 64
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
ACTION_REPEAT = 4

CHANNEL = FRAME_HISTORY #* 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)
GAMMA = 0.99

INIT_EXPLORATION = 1
EXPLORATION_EPOCH_ANNEAL = 0.01
END_EXPLORATION = 0.1

MEMORY_SIZE = 1e6
# NOTE: will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
# Suggest using tcmalloc to manage memory space better.
INIT_MEMORY_SIZE = 5e4
STEP_PER_EPOCH = 10000
EVAL_EPISODE = 50

NUM_ACTIONS = None
DOUBLE = None
DUELING = None
PC_METHOD = None # Pseudo count method

def get_player(viz=False, train=False, dumpdir=None):
    if PC_METHOD and train:
        pl = GymEnv(ENV_NAME, dumpdir=dumpdir, pc_method=PC_METHOD)
    else:
        pl = GymEnv(ENV_NAME, dumpdir=dumpdir)
    def resize(img):
        return cv2.resize(img, IMAGE_SIZE)
    def grey(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = resize(img)
        img = img[:, :, np.newaxis]
        return img
    pl = MapPlayerState(pl, grey)
    #show_images(pl.current_state())

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()
    if not train:
        pl = HistoryFramePlayer(pl, FRAME_HISTORY)
        pl = PreventStuckPlayer(pl, 30, 1)
    pl = LimitLengthPlayer(pl, 40000)
    return pl
common.get_player = get_player  # so that eval functions in common can use the player


class Model(ModelDesc):
    def _get_input_vars(self):
        if NUM_ACTIONS is None:
            p = get_player(); del p
        return [InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
                InputVar(tf.int64, (None,), 'action'),
                InputVar(tf.float32, (None,), 'reward'),
                InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'next_state'),
                InputVar(tf.bool, (None,), 'isOver') ]

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        image = image / 255.0
        with argscope(Conv2D, nl=PReLU.f, use_bias=True):
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)
            # the original arch in Nature DQN
            # l = Conv2D('conv0', image, out_channel=32, kernel_shape=8, stride=4)
            # l = Conv2D('conv1', l, out_channel=64, kernel_shape=4, stride=2)
            # l = Conv2D('conv2', l, out_channel=64, kernel_shape=3)

            l = FullyConnected('fc0', l, 512, nl=lambda x, name: LeakyReLU.f(x, 0.01, name))

        if not DUELING:
            Q = FullyConnected('fct', l, NUM_ACTIONS, nl=tf.identity)
        else:
            V = FullyConnected('fctV', l, 1, nl=tf.identity)
            As = FullyConnected('fctA', l, NUM_ACTIONS, nl=tf.identity)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')

    #TODO: Mix with Monte-Carlo Reward
    def _build_graph(self, inputs):
        state, action, reward, next_state, isOver = inputs
        self.predict_value = self._get_DQN_prediction(state)
        action_onehot = tf.one_hot(action, NUM_ACTIONS, 1.0, 0.0)
        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)    #N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'):
            targetQ_predict_value = self._get_DQN_prediction(next_state)    # NxA

        if not DOUBLE:
            # DQN  # Select the greedy and value from the same target net.
            best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        else:
            # Double-DQN # select the greedy from online net, get value from the target net.
            tf.get_variable_scope().reuse_variables()
            next_predict_value = self._get_DQN_prediction(next_state)
            self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
            predict_onehot = tf.one_hot(self.greedy_choice, NUM_ACTIONS, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * GAMMA * tf.stop_gradient(best_v)

        self.cost = tf.truediv(symbf.huber_loss(target - pred_action_value),
                               tf.cast(BATCH_SIZE, tf.float32), name='cost')

        summary.add_param_summary([('conv.*/W', ['histogram', 'rms']),
                                   ('fc.*/W', ['histogram', 'rms']) ])   # monitor all W

    def update_target_param(self):
        vars = tf.trainable_variables()
        ops = []
        for v in vars:
            target_name = v.op.name
            if target_name.startswith('target'):
                new_name = target_name.replace('target/', '')
                logger.info("{} <- {}".format(target_name, new_name))
                ops.append(v.assign(tf.get_default_graph().get_tensor_by_name(new_name + ':0')))
        return tf.group(*ops, name='update_target_network')

    def get_gradient_processor(self):
        return [MapGradient(lambda grad: \
                tf.clip_by_global_norm([grad], 5)[0][0]),
                SummaryGradient()]

# def get_config():
#     logger.auto_set_dir()
#     M = Model()
#
#     lr = tf.Variable(0.001, trainable=False, name='learning_rate')
#     tf.scalar_summary('learning_rate', lr)
#
#     return TrainConfig(
#         #dataset = ?, # A dataflow object for training
#         optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
#         callbacks=Callbacks([StatPrinter(), ModelSaver(),
#
#                              ]),
#
#         session_config = get_default_sess_config(0.6),  # Tensorflow default session config consume too much resources.
#         model = M,
#         step_per_epoch=STEP_PER_EPOCH,
#     )

def get_config():
    #logger.auto_set_dir()
    #logger.set_logger_dir(os.path.join('train_log', LOG_DIR))
    logger.set_logger_dir(LOG_DIR)
    M = Model()
    #TODO: For count-based model, remove epsilon greedy exploration
    dataset_train = ExpReplay(
            predictor_io_names=(['state'], ['Qvalue']),
            player=get_player(train=True),
            batch_size=BATCH_SIZE,
            memory_size=MEMORY_SIZE,
            init_memory_size=INIT_MEMORY_SIZE,
            exploration=INIT_EXPLORATION,
            end_exploration=END_EXPLORATION,
            exploration_epoch_anneal=EXPLORATION_EPOCH_ANNEAL,
            update_frequency=4,
            reward_clip=(-1, 1),
            history_len=FRAME_HISTORY)

    lr = tf.Variable(0.001, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(), PeriodicCallback(ModelSaver(), 5),
            ScheduledHyperParamSetter('learning_rate',
                [(150, 4e-4), (250, 1e-4), (350, 5e-5)]),
            RunOp(lambda: M.update_target_param()),
            dataset_train,
            PeriodicCallback(Evaluator(EVAL_EPISODE, ['state'], ['Qvalue']), 5),
            #HumanHyperParamSetter('learning_rate', 'hyper.txt'),
            #HumanHyperParamSetter(ObjAttrParam(dataset_train, 'exploration'), 'hyper.txt'),
        ]),
        # save memory for multiprocess evaluator
        session_config=get_default_sess_config(0.6),
        model=M,
        step_per_epoch=STEP_PER_EPOCH,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-l','--load', help='load model')
    parser.add_argument('-e','--env', help='env', required=True)
    parser.add_argument('-t','--task', help='task to perform',
                        choices=['play','eval','train'], default='train')
    parser.add_argument('--double', help='If use double DQN', default='t')
    parser.add_argument('--dueling', help='If use dueling method', default='f')
    parser.add_argument('--logdir', help='output directory', required=True)
    parser.add_argument('--pc', help='pseudo count method', choices=[None, 'joint', 'CTS'], default=None)
    args=parser.parse_args()
    ENV_NAME = args.env
    LOG_DIR  = args.logdir

    if args.double == 't':
        DOUBLE = True
    elif args.double == 'f':
        DOUBLE = False
    else:
        logger.error("double argument must be t or f")
        exit()
    if args.dueling == 't':
        DUELING = True
    elif args.dueling == 'f':
        DUELING = False
    else:
        logger.error("dueling argument must be t or f")
        exit()

    if DOUBLE:
        logger.info("Using Double")
    if DUELING:
        logger.info("Using Dueling")

    # For Pseudo Count Rewards
    PC_METHOD = args.pc
    if PC_METHOD:
        logger.info("Using Pseudo Count method: " + PC_METHOD)
    else:
        logger.info("Don't use Pseudo Count method")

    assert ENV_NAME
    p = get_player(); del p     # set NUM_ACTIONS

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.task != 'train':
        assert args.load is not None

    if args.task != 'train':
        cfg = PredictConfig(
                model=Model(),
                session_init=SaverRestore(args.load),
                input_var_names=['state'],
                output_var_names=['Qvalue'])
        if args.task == 'play':
            play_model(cfg)
        elif args.task == 'eval':
            eval_model_multithread(cfg, EVAL_EPISODE)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        QueueInputTrainer(config).train()
