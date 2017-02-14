#!/usr/bin/env python
# coding: utf-8
# Author: Music
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import deeprl_hw1.lake_envs as lake_env
import gym
import time
import random
import math
import numpy as np
from deeprl_hw1.rl import policy_iteration, value_iteration, value_function_to_policy

def run_random_policy(env):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            env.action_space.sample())
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(1)

    return total_reward, num_steps


def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))


def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))

def generate_random_policy(env, seed=0):
    """
    :return: policy: np.array
                Maps states to actions
    """
    nS = env.nS
    nA = env.nA
    policy = np.array([random.randint(0,nA-1) for i in xrange(nS)])
    #policy = np.array([0 for i in xrange(nS)])
    return policy

def print_values(values):
    l = int(len(values) ** 0.5)
    values = values.reshape((l,l))
    print(values)

def plot_values(values):
    l = int(len(values) ** 0.5)
    values = values.reshape((l,l))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    im = plt.imshow(values, interpolation='none')
    fig.colorbar(im, ax=ax)
    plt.show()

def run_policy(env, gamma, policy):
    state = env.reset()
    total_reward = 0
    num_steps = 0
    discount_factor = 1
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            policy[state])
        total_reward += reward * discount_factor
        discount_factor *= gamma
        num_steps += 1
        if is_terminal:
            break
        state = nextstate
    return total_reward, num_steps

def print_policy(policy, action_names):
    d={'DOWN':'R','RIGHT':'D','UP':'L','LEFT':'U'}
    #d = {'DOWN': 'D', 'RIGHT': 'R', 'UP': 'U', 'LEFT': 'L'}
    l = int(len(policy) ** 0.5)
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, d[action_name])
    str_policy = str_policy.reshape(l, l)
    print(str_policy)


def main():
    # create the environment
    # env = gym.make('FrozenLake-v0')
    # uncomment next line to try the deterministic version
    gamma = 0.9
    #env = gym.make('Deterministic-4x4-FrozenLake-v0')
    #env = gym.make('Deterministic-8x8-FrozenLake-v0')
    #env = gym.make('Stochastic-4x4-FrozenLake-v0')
    #env = gym.make('Stochastic-8x8-FrozenLake-v0')
    env = gym.make('Deterministic-4x4-neg-reward-FrozenLake-v0')
    action_names = lake_env.action_names
    policy = generate_random_policy(env)
    #print_policy(policy, action_names)
    env.render()

    # input('Hit enter to run policy iteration...')
    # start = time.time()
    # policy, value_func, num_policy_imp, num_value_iters = policy_iteration(env, gamma)
    # end   = time.time()
    # print("Execute time", end-start)
    # print_policy(policy, action_names)
    # print_values(value_func)
    # print("The number of policy improvements: %d"%(num_policy_imp))
    # print("The number of value iterations: %d" % (num_value_iters))
    # #print(run_policy(env, gamma, policy))
    # #plot_values(value_func)

    input('Hit enter to run value iteration...')
    start = time.time()
    values, num_value_iters = value_iteration(env, gamma)
    policy = value_function_to_policy(env, gamma, values)
    end = time.time()
    print("Execute time", end - start)
    print_policy(policy, action_names)
    print_values(values)
    plot_values(values)
    print("The number of value iterations: %d" % (num_value_iters))
    #total_reward = 0
    #for i in xrange(100000):
    #    reward, step = run_policy(env, gamma, policy)
    #    total_reward += reward
    #print(total_reward/100000)
    #print(value_func, num_policy_imp, num_value_iters)
    #values, i = evaluate_policy(env, gamma, policy, tol=10e-3)
    #print(values,i)
    #policy_changed, policy = improve_policy(env, gamma, values, policy)
    #print(policy_changed,policy)




    # print_env_info(env)
    # print_model_info(env, 0, lake_env.DOWN)
    # print_model_info(env, 1, lake_env.DOWN)
    # print_model_info(env, 14, lake_env.RIGHT)
    #
    # input('Hit enter to run a random policy...')
    #
    # total_reward, num_steps = run_random_policy(env)
    # print('Agent received total reward of: %f' % total_reward)
    # print('Agent took %d steps' % num_steps)


if __name__ == '__main__':
    main()
