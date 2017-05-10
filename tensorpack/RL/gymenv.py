#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gymenv.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>, Music Lee <yuezhanl@andrew.cmu.edu>


import time
from pseudocount import *
from ..utils import logger
import sys
try:
    import gym
    # TODO
    #gym.undo_logger_setup()
    # https://github.com/openai/gym/pull/199
    # not sure does it cause other problems
    __all__ = ['GymEnv']
except ImportError:
    logger.warn("Cannot import gym. GymEnv won't be available.")
    __all__ = []

import threading

from ..utils.fs import *
from ..utils.stat import *
from .envbase import RLEnvironment, DiscreteActionSpace
from collections import deque

_ALE_LOCK = threading.Lock()

class GymEnv(RLEnvironment):
    """
    An OpenAI/gym wrapper. Can optionally auto restart.
    """
    def __init__(self, name, pc_method=None, pc_mult=None, pc_thre=None, pc_time=None, dumpdir=None, viz=False, auto_restart=True, feature=None, pc_action=False, pc_downsample_value=None, pc_clean=False):
        # pc_method: Pseudo-count exploration method
        self.pc_method = pc_method
        self.multiplier = 1
        self.pc_mult = pc_mult
        self.pc_downsample_value = pc_downsample_value
        self.pc_clean = pc_clean
        if pc_method:
            self.pc = PC(pc_method, pc_downsample_value)
            if self.pc_mult:
                self.pc_repeat_time = 0
                self.pc_max_repeat_time = pc_time
                self.pc_thre = pc_thre
        with _ALE_LOCK:
            self.gymenv = gym.make(name)
        if dumpdir:
            mkdir_p(dumpdir)
            self.gymenv.monitor.start(dumpdir)
        self.use_dir = dumpdir

        self.reset_stat()
        self.rwd_counter = StatCounter()
        self.restart_episode()
        self.auto_restart = auto_restart
        self.viz = viz
        self.feature = feature
        self.pc_action = pc_action
        self.step = 0

    def original_current_state(self):
        return self._ob

    def restart_episode(self):
        self.rwd_counter.reset()
        self._ob = self.gymenv.reset()

    def finish_episode(self):
        if self.use_dir is not None:
            self.gymenv.monitor.flush()
        self.stats['score'].append(self.rwd_counter.sum)

    def current_state(self):
        if self.viz:
            self.gymenv.render()
            time.sleep(self.viz)
        return self._ob

    def action(self, act):
        if self.feature and type(act) == list:
            # use feature and pass feature values from master.
            (act, feature) = act
        old_ob = np.copy(self._ob)
        self._ob, r, isOver, info = self.gymenv.step(act)
        if self.pc_method:
            # Clear counter every 10 epoch
            self.step += 1
            if self.pc_clean:
                if self.step == 60000:
                    self.step = 0
                    self.pc.clear()
            if not self.feature:
                if not self.pc_action:
                    pc_reward = self.pc.pc_reward(self._ob)
                else:
                    pc_reward = self.pc.pc_reward_with_act(old_ob, act)
            else:
                pc_reward = self.pc.pc_reward_feature(feature)
            pc_reward = pc_reward * self.multiplier
            r += pc_reward
            if self.pc_mult:
                if pc_reward < self.pc_thre:
                    self.pc_repeat_time += 1
                else:
                    self.pc_repeat_time = 0
                if self.pc_repeat_time >= self.pc_max_repeat_time:
                    self.multiplier *= self.pc_mult
                    self.pc_repeat_time = 0
                    logger.info('Multiplier for pc reward is getting bigger. Multiplier=' + str(self.multiplier))
            #sys.stderr.write(str(r)+'\n')
        self.rwd_counter.feed(r)
        if isOver:
            self.finish_episode()
            if self.auto_restart:
                self.restart_episode()
        return r, isOver

    def get_action_space(self):
        spc = self.gymenv.action_space
        assert isinstance(spc, gym.spaces.discrete.Discrete)
        return DiscreteActionSpace(spc.n)

    def finish(self):
        self.gymenv.monitor.close()

if __name__ == '__main__':
    env = GymEnv('Breakout-v0', viz=0.1)
    num = env.get_action_space().num_actions()

    from ..utils import *
    rng = get_rng(num)
    while True:
        act = rng.choice(range(num))
        #print act
        r, o = env.action(act)
        env.current_state()
        if r != 0 or o:
            print(r, o)
