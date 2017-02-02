#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from tensorpack.predict.base import OfflinePredictor
import gym
import numpy as np

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
CHANNEL = FRAME_HISTORY# * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

NUM_ACTIONS = None
ENV_NAME = None
DOUBLE = None
DUELING = None

from common import play_one_episode, get_predict_func

def get_player(dumpdir=None):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir, auto_restart=False)
    def resize(img):
        return cv2.resize(img, IMAGE_SIZE)
    def grey(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = resize(img)
        img = img[:, :, np.newaxis] / 255.0
        return img
    pl = MapPlayerState(pl, grey)


    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()

    pl = HistoryFramePlayer(pl, FRAME_HISTORY)
    #show_images(pl.current_state())
    return pl

class Model(ModelDesc):
    def _get_input_vars(self):
        assert NUM_ACTIONS is not None
        return [InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'state')]#,
                #InputVar(tf.int32, (None,), 'action'),
                #InputVar(tf.float32, (None,), 'futurereward') ]

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        #image = image / 255.0
        with argscope(Conv2D, nl=PReLU.f, use_bias=True):
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

            l = FullyConnected('fc0', l, 512, nl=lambda x, name: LeakyReLU.f(x, 0.01, name))
            # the original arch
            #.Conv2D('conv0', image, out_channel=32, kernel_shape=8, stride=4)
            #.Conv2D('conv1', out_channel=64, kernel_shape=4, stride=2)
            #.Conv2D('conv2', out_channel=64, kernel_shape=3)

        if not DUELING:
            Q = FullyConnected('fct', l, NUM_ACTIONS, nl=tf.identity)
        else:
            V = FullyConnected('fctV', l, 1, nl=tf.identity)
            As = FullyConnected('fctA', l, NUM_ACTIONS, nl=tf.identity)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')

    def _build_graph(self, inputs):
        state = inputs[0]
        #state, action, futurereward = inputs
        self.Qvalue = self._get_DQN_prediction(state)
        max_Qvalue = tf.reduce_max(self.Qvalue, 1)
        saliency = tf.gradients(max_Qvalue, state)[0]
        self.saliency = tf.identity(saliency, name='saliency')


def run(cfg, s_cfg, output):
    player = get_player(dumpdir=output)
    predfunc = OfflinePredictor(cfg)
    s_func   = OfflinePredictor(s_cfg)
    timestep = 0
    #sa = Saliency_Analyzor('../obj/MsPacman-v0')
    while True:
        timestep += 1
        s = player.current_state()
        s0 = player.original_current_state()
        act = predfunc([[s]])[0][0].argmax()
        saliency = s_func([[s]])[0][0]
        r, isOver = player.action(act)
        show(s, saliency, act, timestep, output, last=True, save=True)
        #show_large(s0, saliency, act, timestep, output, save=True, save_npy=False)
        #print r, act
        if timestep % 50 == 0:
            print timestep
        if isOver:
            return

def show_large(s, saliency, act, timestep, output, save=False, save_npy=False, analyzor=None):
    # Show the pictures of original resolution of the game play
    # Convert the 84*84 saliency maps to 210 * 160 resolution
    import matplotlib.pyplot as plt
    # Get the saliency map for the last frame in the history (The current frame)
    saliency = saliency[:,:,3]
    saliency = cv2.resize(saliency, (160,210))
    if save_npy:
        np.save(output+"/state%d"%timestep, s)
        np.save(output+"/saliency%d"%timestep, saliency)
    # object saliency maps
    if analyzor:
        obj_sals = analyzor.object_saliencies(s, saliency)
        obj_sals = analyzor.object_saliencies_filter(obj_sals)
        s = analyzor.saliency_image(s, obj_sals)
    plt.subplot(211)
    plt.axis('off')
    fig = plt.imshow(s, aspect='equal')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.subplot(212)
    plt.title('action:' + str(act))
    plt.axis('off')
    fig = plt.imshow(saliency, aspect='equal')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig(output + "/file%04d.png" % timestep, bbox_inches='tight', pad_inches = 0)
    else:
        plt.show()

def show(s, saliency, act, timestep, output, last=False, save=False):
    import matplotlib.pyplot as plt
    for i in xrange(s.shape[2]):
        if last:
            if i != s.shape[2] - 1:
                continue
        plt.subplot(211)
        plt.axis('off')
        fig = plt.imshow(s[:, :, i], cmap='gray', aspect='equal', interpolation='nearest')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.subplot(212)
        plt.title('action:' + str(act))
        plt.axis('off')
        fig = plt.imshow(saliency[:,:,i], cmap='gray', aspect='equal', interpolation='nearest')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        if save:
            plt.savefig(output + "/file%04d.png" % timestep, bbox_inches='tight', pad_inches = 0)
        else:
            plt.show()

def run_submission(cfg, output, nr):
    player = get_player(dumpdir=output)
    predfunc = get_predict_func(cfg)
    for k in range(nr):
        if k != 0:
            player.restart_episode()
        score = play_one_episode(player, predfunc)#, task='save_image')
        print("Total:", score)

def do_submit(output, api_key):
    gym.upload(output, api_key=api_key)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model', required=True)
    parser.add_argument('--env', help='environment name', required=True)
    parser.add_argument('--episode', help='number of episodes to run',
            type=int, default=100)
    parser.add_argument('--output', help='output directory', default='gym-submit')
    parser.add_argument('--double', help='If use double DQN', default='t')
    parser.add_argument('--dueling', help='If use dueling method', default='f')
    parser.add_argument('--api', help='gym api key')
    #parser.add_argument('--task', help='task to perform', choices=['gym','sample'], default='gym')
    args = parser.parse_args()

    ENV_NAME = args.env
    if args.double == 't':
        DOUBLE = True
    elif args.double == 'f':
        DOUBLE = False
    else:
        logger.error("double argument must be t or f")
    if args.dueling == 't':
        DUELING = True
    elif args.dueling == 'f':
        DUELING = False
    else:
        logger.error("dueling argument must be t or f")

    if DOUBLE:
        logger.info("Using Double")
    if DUELING:
        logger.info("Using Dueling")

    assert ENV_NAME
    logger.info("Environment Name: {}".format(ENV_NAME))
    p = get_player(); del p    # set NUM_ACTIONS

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = PredictConfig(
            model=Model(),
            session_init=SaverRestore(args.load),
            input_var_names=['state'],
            output_var_names=['Qvalue'])

    s_cfg = PredictConfig(
            model=Model(),
            session_init=SaverRestore(args.load),
            input_var_names=['state'],
            output_var_names=['saliency'])

    run(cfg, s_cfg, args.output)
    #run_submission(cfg, args.output, args.episode)
    #do_submit(args.output, args.api)
