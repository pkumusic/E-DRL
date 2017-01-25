#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: envbase.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from abc import abstractmethod, ABCMeta
from collections import defaultdict
import random
from ..utils import get_rng


__all__ = ['RLEnvironment', 'NaiveRLEnvironment', 'ProxyPlayer',
           'DiscreteActionSpace']

STEP = 0

class RLEnvironment(object):
    __meta__ = ABCMeta

    def __init__(self):
        self.reset_stat()

    @abstractmethod
    def current_state(self):
        """
        Observe, return a state representation
        """

    @abstractmethod
    def action(self, act):
        """
        Perform an action. Will automatically start a new episode if isOver==True
        :param act: the action
        :returns: (reward, isOver)
        """

    def restart_episode(self):
        """ Start a new episode, even if the current hasn't ended """
        raise NotImplementedError()

    def finish_episode(self):
        """ get called when an episode finished"""
        pass

    def get_action_space(self):
        """ return an `ActionSpace` instance"""
        raise NotImplementedError()

    def reset_stat(self):
        """ reset all statistics counter"""
        self.stats = defaultdict(list)

    def play_one_episode(self, func, stat='score', task=None):
        """ play one episode for eval.
            :param func: call with the state and return an action
            :param stat: a key or list of keys in stats
            :returns: the stat(s) after running this episode
        """
        if not isinstance(stat, list):
            stat = [stat]
        while True:
            s = self.current_state()
            if task == 'save_image': # Used in DQN-gym-run.py to sample images by loading model
                import cv2
                import matplotlib.pyplot as plt
                import pylab
                import numpy as np
                global STEP
                STEP += 1
                file_name = '../obj/MsPacman-v0-sample/' + str(STEP)
                img  = self.original_current_state()
                np.save(file_name, img)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #plt.imshow(img)#, cmap=pylab.gray())
                #plt.savefig(file_name)

            act = func(s)
            r, isOver = self.action(act)
            #print r
            if isOver:
                if task == 'save_image':
                    exit()
                s = [self.stats[k] for k in stat]
                self.reset_stat()
                return s if len(s) > 1 else s[0]

class ActionSpace(object):
    def __init__(self):
        self.rng = get_rng(self)

    @abstractmethod
    def sample(self):
        pass

    def num_actions(self):
        raise NotImplementedError()

class DiscreteActionSpace(ActionSpace):
    def __init__(self, num):
        super(DiscreteActionSpace, self).__init__()
        self.num = num

    def sample(self):
        return self.rng.randint(self.num)

    def num_actions(self):
        return self.num

    def __repr__(self):
        return "DiscreteActionSpace({})".format(self.num)

    def __str__(self):
        return "DiscreteActionSpace({})".format(self.num)

class NaiveRLEnvironment(RLEnvironment):
    """ for testing only"""
    def __init__(self):
        self.k = 0
    def current_state(self):
        self.k += 1
        return self.k
    def action(self, act):
        self.k = act
        return (self.k, self.k > 10)

class ProxyPlayer(RLEnvironment):
    """ Serve as a proxy another player """
    def __init__(self, player):
        self.player = player

    def reset_stat(self):
        self.player.reset_stat()

    def current_state(self):
        return self.player.current_state()

    def action(self, act):
        return self.player.action(act)

    def original_current_state(self):
        return self.player.original_current_state()

    @property
    def stats(self):
        return self.player.stats

    def restart_episode(self):
        self.player.restart_episode()

    def finish_episode(self):
        self.player.finish_episode()

    def get_action_space(self):
        return self.player.get_action_space()
