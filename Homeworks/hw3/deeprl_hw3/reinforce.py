import gym
import numpy as np


def get_total_reward(env, model):
    """compute total reward

    Parameters
    ----------
    env: gym.core.Env
      The environment. 
    model: (your action model, which can be anything)

    Returns
    -------
    total_reward: float
    """
    return 0.0


def choose_action(model, observation):
    """choose the action 

    Parameters
    ----------
    model: (your action model, which can be anything)
    observation: given observation

    Returns
    -------
    p: float 
        probability of action 1
    action: int
        the action you choose
    """
    return .5, 0


def reinforce(env):
    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment

    Returns
    -------
    total_reward: float
    """
    return 0
