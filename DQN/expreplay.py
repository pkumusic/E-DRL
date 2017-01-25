#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DL Project for 10807
# Authors: Music, Tian, Jing
# This file is trying to understand expreplay

from tensorpack.RL.expreplay import ExpReplay
from tensorpack.RL.gymenv import GymEnv
from tensorpack.RL.common import MapPlayerState, PreventStuckPlayer, LimitLengthPlayer
from tensorpack.RL.history import HistoryFramePlayer
import cv2
import numpy as np
import sys

ENV_NAME = 'Freeway-v0'
FRAME_HISTORY = 4

def get_player(viz=False, train=False, dumpdir=None):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir)
    #def func(img):
    #    return cv2.resize(img, IMAGE_SIZE[::-1])
    #pl = MapPlayerState(pl, func)

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()
    if not train:
        pass
        pl = HistoryFramePlayer(pl, FRAME_HISTORY)
        #pl = PreventStuckPlayer(pl, 30, 1) #TODO: I think we don't need this in freeway. Is any bug in this code? didn't see repeated actions.
    pl = LimitLengthPlayer(pl, 40000)
    return pl


if __name__ == '__main__':
    import sys
    predictor = lambda x: np.array([1,1,1,1])
    player = get_player(train=True)
    E = ExpReplay(predictor,
            player=player,
            init_memory_size=100,
            history_len=1)
    E._init_memory()

    for [state, action, reward, next_state, isOver] in E.get_data():
        E.before_train()
        print state.shape

