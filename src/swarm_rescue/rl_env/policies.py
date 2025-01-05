import functools
from typing import Dict
import numpy as np
import pickle
import torch
import os

from solutions.utils.constants import FRONTIER_COUNT, ACTION_SPACE, FRONTIER_FEATURES,OTHER_FRONTIER_INPUTS

def get_obs(input:np.ndarray)->Dict[str, np.ndarray]:
    keys = [
        'size',
        'count',
        'distance',
        'repulsion_angle',
        'direction_angle',
        'distance_last'
    ]
    keys.sort()
    obs = {}
    for k in range(FRONTIER_COUNT):
        obs[keys[k]] = input[k*FRONTIER_FEATURES:(k+1)*FRONTIER_FEATURES]
    return obs


def deterministic_policy(input):
    coef = {
        'size': 1,
        'count': 1,
        'distance': 2,
        'repulsion_angle': 4,
        'direction_angle': 1,
        'distance_last': 1
    }
    output = np.zeros(FRONTIER_COUNT)
    obs = get_obs(input)
    for feature in coef:
        output += coef[feature] * obs[feature]
    return output


def epsilon_greedy_policy(input,eps):
    if np.random.random() < eps:
        output = np.zeros(ACTION_SPACE)
        output[np.random.randint(0,ACTION_SPACE)] = 1
        return output
    else:
        return deterministic_policy(input)

def epsilon_greedy_wrapper(eps):
    return functools.partial(epsilon_greedy_policy, eps=eps)


def get_custom_policy(agent_file, epsilon):

    agent = torch.load(agent_file, map_location=torch.device('cpu'), weights_only=False)

    def policy(input):
        if np.random.random() < epsilon:
            output = np.zeros(ACTION_SPACE)
            output[np.random.randint(0, ACTION_SPACE)] = 1
            return output
        else:
            with torch.no_grad():
                input_tensor = torch.tensor(input, dtype=torch.float32)
                output_tensor = agent(input_tensor)
                output = output_tensor.detach().cpu().numpy()
            return output

    return policy


def get_multi_input_policy(agent_file, epsilon):
    policy = get_custom_policy(agent_file, epsilon)

    def new_policy(x):
        output = np.zeros(ACTION_SPACE)
        for i in range(ACTION_SPACE):
            input = np.zeros(FRONTIER_FEATURES + OTHER_FRONTIER_INPUTS)
            for j in range(ACTION_SPACE):
                input[j] = x[j * FRONTIER_FEATURES + i]
            for j in range(OTHER_FRONTIER_INPUTS):
                input[FRONTIER_FEATURES + j] = x[ACTION_SPACE * FRONTIER_FEATURES + j]
            output[i] = policy(input)[0]
        return output

    return new_policy