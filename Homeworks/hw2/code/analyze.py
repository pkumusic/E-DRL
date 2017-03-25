#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DL Project for 10807
# Author: Music, Tian, Jing

# Analyze the model results
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import spline


def draw_performance_mean(filePath, label):
    data = json.load(open(filePath, 'r'))
    global_steps = []
    exp_mean_scores = []
    print filePath
    for d in data:
        if d['global_step'] > 3*1e6:
            continue
        #if 'expreplay/mean_score' in d:
        #    exp_mean_scores.append(d['expreplay/mean_score'])
        #    global_steps.append(d['global_step'])
        elif 'mean_score' in d:
            exp_mean_scores.append(d['mean_score'])
            global_steps.append(d['global_step'])
        else:
            pass
    global_steps, exp_mean_scores = np.array(global_steps), np.array(exp_mean_scores)
    x_smooth = np.linspace(global_steps.min(), global_steps.max(),50)
    y_smooth = spline(global_steps, exp_mean_scores, x_smooth)
    if label.startswith('O'):
        marker = '^'
    else:
        marker = 'o'
    #plt.plot(global_steps, exp_mean_scores, label=label, marker=marker)
    plt.plot(x_smooth, y_smooth, label=label, marker=marker)


def draw_performance_max(filePath, label):
    data = json.load(open(filePath, 'r'))
    global_steps = []
    exp_mean_scores = []
    print filePath
    for d in data:
        if 'expreplay/max_score' in d:
            exp_mean_scores.append(d['expreplay/max_score'])
            global_steps.append(d['global_step'])
        elif 'max_score' in d:
            exp_mean_scores.append(d['max_score'])
            global_steps.append(d['global_step'])
        else:
            pass
    global_steps, exp_mean_scores = np.array(global_steps), np.array(exp_mean_scores)
    x_smooth = np.linspace(global_steps.min(), global_steps.max(),50)
    y_smooth = spline(global_steps, exp_mean_scores, x_smooth)
    if label.startswith('O'):
        marker = '^'
    else:
        marker = 'o'
    plt.plot(x_smooth, y_smooth, label=label, marker=marker)

if __name__ == '__main__':
    log_path = 'results'
    dirs = glob.glob(log_path + '/*')
    files = map(lambda x:x+'/stat.json', dirs)
    labels = map(lambda x:x.split('/')[1], dirs)
    method = 'mean'
    for i in xrange(len(files)):
        if method == 'mean':
            draw_performance_mean(files[i], labels[i])
        elif method == 'max':
            draw_performance_max(files[i], labels[i])
    #plt.legend()
    plt.legend(loc=2)
    plt.xlabel('Number of iterations')
    if method == 'mean':
        plt.ylabel('Mean score over 50 plays')
        plt.title("Average Mean Score Played by DRL Agents in Game SpaceInvaders")
    elif method == 'max':
        plt.ylabel('Max score over 50 plays')
        plt.title("Max Score Played by DRL Agents in Game SpaceInvaders")
    plt.show()

