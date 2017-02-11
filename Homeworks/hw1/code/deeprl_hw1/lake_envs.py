# coding: utf-8
"""Defines some frozen lake maps."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym.envs.toy_text.frozen_lake import LEFT, RIGHT, DOWN, UP
from gym.envs.toy_text import frozen_lake, discrete

from gym.envs.registration import register

action_names = {LEFT: 'LEFT', RIGHT: 'RIGHT', DOWN: 'DOWN', UP: 'UP'}

register(
    id='Deterministic-4x4-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False})

register(
    id='Deterministic-8x8-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '8x8',
            'is_slippery': False})

register(
    id='Stochastic-4x4-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': True})

register(
    id='Stochastic-8x8-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '8x8',
            'is_slippery': True})


class NegRewardFrozenLake(frozen_lake.FrozenLakeEnv):
    def __init__(self, **kwargs):
        super(NegRewardFrozenLake, self).__init__(**kwargs)

        # modify the rewards
        for state in range(self.nS):
            for action in range(self.nA):
                new_transitions = []
                for (prob, nextstate, _, is_terminal) in self.P[state][action]:
                    row = nextstate // self.ncol
                    col = nextstate - row * self.ncol
                    tile_type = self.desc[row, col]
                    if tile_type == b'F' or tile_type == b'S':
                        reward = -1
                    elif tile_type == b'G':
                        reward = 1
                    else:
                        reward = 0

                    new_transitions.append(
                        (prob, nextstate, reward, is_terminal))
                self.P[state][action] = new_transitions


register(
    id='Deterministic-4x4-neg-reward-FrozenLake-v0',
    entry_point='deeprl_hw1.lake_envs:NegRewardFrozenLake',
    kwargs={'map_name': '4x4',
            'is_slippery': False})

register(
    id='Stochastic-4x4-neg-reward-FrozenLake-v0',
    entry_point='deeprl_hw1.lake_envs:NegRewardFrozenLake',
    kwargs={'map_name': '4x4',
            'is_slippery': True})

register(
    id='Deterministic-8x8-neg-reward-FrozenLake-v0',
    entry_point='deeprl_hw1.lake_envs:NegRewardFrozenLake',
    kwargs={'map_name': '8x8',
            'is_slippery': False})

register(
    id='Stochastic-8x8-neg-reward-FrozenLake-v0',
    entry_point='deeprl_hw1.lake_envs:NegRewardFrozenLake',
    kwargs={'map_name': '8x8',
            'is_slippery': True})
