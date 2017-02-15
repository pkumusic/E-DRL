#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import deeprl_hw1.queue_envs as queue_envs
import gym
import time
if __name__ == "__main__":
    env = gym.make('Queue-1-v0')
    env.step(0)
