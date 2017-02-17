# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import gym
from gym import Env, spaces
from gym.envs.registration import register
import numpy as np

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

class QueueEnv(Env):
    """Implement the Queue environment from problem 3.

    Parameters
    ----------
    p1: float
      Value between [0, 1]. The probability of queue 1 receiving a new item.
    p2: float
      Value between [0, 1]. The probability of queue 2 receiving a new item.
    p3: float
      Value between [0, 1]. The probability of queue 3 receiving a new item.

    Attributes
    ----------
    nS: number of states
    nA: number of actions
    P: environment model
    """
    metadata = {'render.modes': ['human']}

    SWITCH_TO_1 = 0
    SWITCH_TO_2 = 1
    SWITCH_TO_3 = 2
    SERVICE_QUEUE = 3

    def __init__(self, p1, p2, p3):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(
            [(1, 3), (0, 5), (0, 5), (0, 5)])
        self.nS = 3 * 6 * 6 * 6
        self.nA = 4
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.s = (1, 0, 0, 0)
        self._seed()
        self._reset()
        # define P here P[s][a] = [(prob, nextstate, reward, is_terminal)...]
        self.P = {}
        for pointer in xrange(1,4):
            for num1 in xrange(6):
                for num2 in xrange(6):
                    for num3 in xrange(6):
                        state = (pointer, num1, num2, num3)
                        self.P[state] = {}
                        for a in xrange(4):
                            self.P[state][a] = self.query_model(state, a)


    def _reset(self):
        """Reset the environment.

        The server should always start on Queue 1.

        Returns
        -------
        (int, int, int, int)
          A tuple representing the current state with meanings
          (current queue, num items in 1, num items in 2, num items in
          3).
        """
        return (1, 0, 0, 0)

    def _step(self, action):
        """Execute the specified action.

        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """
        # Action
        # (cur, num1, num2, num3) = self.s
        # reward = 0
        # if action == self.SWITCH_TO_1:
        #     cur = 1
        # elif action == self.SWITCH_TO_2:
        #     cur = 2
        # elif action == self.SWITCH_TO_3:
        #     cur = 3
        # elif action == self.SERVICE_QUEUE:
        #     if self.s[cur] > 0:
        #         reward = 1
        #         if cur == 1:
        #             num1 -= 1
        #         if cur == 2:
        #             num2 -= 1
        #         if cur == 3:
        #             num3 -= 1
        # # After action.
        # add1 = np.random.binomial(1, self.p1)
        # add2 = np.random.binomial(1, self.p2)
        # add3 = np.random.binomial(1, self.p3)
        # if add1 and num1 < 5:
        #     num1 += 1
        # if add2 and num2 < 5:
        #     num2 += 1
        # if add3 and num3 < 5:
        #     num3 += 1
        # self.s = (cur, num1, num2, num3)
        #return self.s, reward, False, None
        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        return (s, r, False, d)



    def _render(self, mode='human', close=False):
        print(self.s)


    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        self.np_random = np.random.RandomState()
        self.np_random.seed(seed)
        return seed

    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.

        This should be in the same format at the provided environments
        in section 2.

        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.

        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """
        (cur, num1, num2, num3) = state
        reward = 0
        if action == QueueEnv.SWITCH_TO_1:
            cur = 1
        if action == QueueEnv.SWITCH_TO_2:
            cur = 2
        if action == QueueEnv.SWITCH_TO_3:
            cur = 3
        if action == QueueEnv.SERVICE_QUEUE:
            if state[cur] > 0:
                reward = 1
                if cur == 1:
                    num1 -= 1
                if cur == 2:
                    num2 -= 1
                if cur == 3:
                    num3 -= 1
        ans = []
        for p1, add1 in ((self.p1, 1), (1-self.p1, 0)):
            for p2, add2 in ((self.p2, 1), (1-self.p2, 0)):
                for p3, add3 in ((self.p3, 1), (1 - self.p3, 0)):
                    p = p1 * p2 * p3
                    new_num1, new_num2, new_num3 = num1, num2, num3
                    if num1 < 5 and add1:
                        new_num1 = num1 + 1
                    if num2 < 5 and add2:
                        new_num2 = num2 + 1
                    if num3 < 5 and add3:
                        new_num3 = num3 + 1
                    ans.append((p, (cur, new_num1, new_num2, new_num3), reward, False))
        return ans

    def get_action_name(self, action):
        if action == QueueEnv.SERVICE_QUEUE:
            return 'SERVICE_QUEUE'
        elif action == QueueEnv.SWITCH_TO_1:
            return 'SWITCH_TO_1'
        elif action == QueueEnv.SWITCH_TO_2:
            return 'SWITCH_TO_2'
        elif action == QueueEnv.SWITCH_TO_3:
            return 'SWITCH_TO_3'
        return 'UNKNOWN'


register(
    id='Queue-1-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .9,
            'p3': .1})

register(
    id='Queue-2-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .1,
            'p3': .1})

