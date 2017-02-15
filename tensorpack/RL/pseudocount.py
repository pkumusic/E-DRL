#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: pseudocount.py
# Author: Music Li <yuezhanl@andrew.cmu.edu>
from __future__ import division
import cv2
import numpy as np
FRSIZE = 42
MAXVAL = 255                # original max value for a state
MAX_DOWNSAMPLED_VAL = 255   # downsampled max value for a state. 8 in the paper.
class PC():
    # class for process with pseudo count rewards
    def __init__(self, method):
        # initialize
        self.method = method
        self.flat_pixel_counter = np.zeros((FRSIZE*FRSIZE, MAX_DOWNSAMPLED_VAL+1)) # Counter for each (pos1, pos2, val), used for joint method
        self.total_num_states = 0  # total number of seen states



    def pc_reward(self, state):
        """
        The final API used by others.
        Given an state, return back the final pseudo count reward
        :return:
        """
        state = self.preprocess(state)
        pc_reward = self.add(state)

        return pc_reward

    def preprocess(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, (FRSIZE, FRSIZE))
        state = np.uint8(state / MAXVAL * MAX_DOWNSAMPLED_VAL)
        return state

    def add(self, state):
        if self.method == 'joint':
            # Model each pixel as independent pixels.
            # p = (c1/n) * (c2/n) * ... * (cn/n)
            # pp = (c1+1)/(n+1) * (c2+1)/(n+1) ...
            # N = (p/pp * (1-pp))/(1-p/pp) ~= (p/pp) / (1-p/pp)
            state = np.reshape(state, (FRSIZE*FRSIZE))
            if self.total_num_states > 0:
                nr = (self.total_num_states + 1.0) / self.total_num_states
                pixel_count = self.flat_pixel_counter[range(FRSIZE*FRSIZE), state]
                self.flat_pixel_counter[range(FRSIZE*FRSIZE), state] += 1
                p_over_pp = np.prod(nr*pixel_count/(1.0+pixel_count))
                denominator = 1.0 - p_over_pp
                if denominator <= 0.0:
                    print "psc_add_image: dominator <= 0.0 : dominator=", denominator
                    denominator = 1.0e-20
                pc_count = p_over_pp / denominator
                pc_reward = self.count2reward(pc_count)
            else:
                pc_count = 0.0
                pc_reward = self.count2reward(pc_count)
            self.total_num_states += 1
        return pc_reward

    def count2reward(self, count, beta=0.05, alpha=0.01, power=-0.5):
        # r = beta (N + alpha)^power
        return beta * ((count+alpha) ** power)

