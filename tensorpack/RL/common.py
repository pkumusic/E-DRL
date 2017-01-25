#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Project for 10807


import numpy as np
from collections import deque
from .envbase import ProxyPlayer

__all__ = ['PreventStuckPlayer', 'LimitLengthPlayer', 'AutoRestartPlayer',
        'MapPlayerState']

class PreventStuckPlayer(ProxyPlayer):
    """ Prevent the player from getting stuck (repeating a no-op)
    by inserting a different action. Useful in games such as Atari Breakout
    where the agent needs to press the 'start' button to start playing.
    """
    # TODO hash the state as well?
    def __init__(self, player, nr_repeat, action):
        """
        :param nr_repeat: trigger the 'action' after this many of repeated action
        :param action: the action to be triggered to get out of stuck
        Does auto-reset, but doesn't auto-restart the underlying player.
        """
        super(PreventStuckPlayer, self).__init__(player)
        self.act_que = deque(maxlen=nr_repeat)
        self.trigger_action = action

    def action(self, act):
        self.act_que.append(act)
        if self.act_que.count(self.act_que[0]) == self.act_que.maxlen:
            act = self.trigger_action
        r, isOver = self.player.action(act)
        if isOver:
            self.act_que.clear()
        return (r, isOver)

    def restart_episode(self):
        super(PreventStuckPlayer, self).restart_episode()
        self.act_que.clear()

class LimitLengthPlayer(ProxyPlayer):
    """ Limit the total number of actions in an episode.
        Will auto restart the underlying player on timeout
    """
    def __init__(self, player, limit):
        super(LimitLengthPlayer, self).__init__(player)
        self.limit = limit
        self.cnt = 0

    def action(self, act):
        r, isOver = self.player.action(act)
        self.cnt += 1
        if self.cnt >= self.limit:
            isOver = True
            self.finish_episode()
            self.restart_episode()
        if isOver:
            self.cnt = 0
        return (r, isOver)

    def restart_episode(self):
        self.player.restart_episode()
        self.cnt = 0

class AutoRestartPlayer(ProxyPlayer):
    """ Auto-restart the player on episode ends,
        in case some player wasn't designed to do so. """
    def action(self, act):
        r, isOver = self.player.action(act)
        if isOver:
            self.player.finish_episode()
            self.player.restart_episode()
        return r, isOver

class MapPlayerState(ProxyPlayer):
    def __init__(self, player, func):
        super(MapPlayerState, self).__init__(player)
        self.func = func

    def current_state(self):
        return self.func(self.player.current_state())

class ObjectSensitivePlayer(ProxyPlayer):
    def __init__(self, player, templateMatcher, method, func):
        super(ObjectSensitivePlayer, self).__init__(player)
        self.templateMatcher = templateMatcher
        self.method = method
        self.func = func

    def current_state(self):
        img = self.player.original_current_state()
        obj_areas = self.templateMatcher.match_all_objects(img)
        obj_images = self.templateMatcher.process_image(img, obj_areas, self.method)
        if self.method == 'add_input_separate':
            obj_images = self.func(obj_images)
            state = self.player.current_state()
            new_state = np.concatenate((state, obj_images), axis=2)
            return new_state
        elif self.method == 'swap_input_separate':
            obj_images = self.func(obj_images)
            return obj_images
        elif self.method == 'swap_input_combine':
            obj_images /= float(len(self.templateMatcher.index2obj))
            obj_images = self.func(obj_images)
            obj_images = obj_images[:, :, np.newaxis]
            return obj_images
        elif self.method == 'add_input_combine':
            obj_images /= float(len(self.templateMatcher.index2obj))
            obj_images = self.func(obj_images)
            obj_images = obj_images[:, :, np.newaxis]
            state = self.player.current_state()
            new_state = np.concatenate((state, obj_images), axis=2)
            return new_state


def show_images(img, last=False):
    import matplotlib.pyplot as plt
    for i in xrange(img.shape[2]):
        if last:
            if i != img.shape[2]-1:
                continue
        plt.imshow(img[:,:,i])
        plt.show()
