from typing import Dict

import numpy as np

from solutions.utils.constants import ACTION_SPACE, FRONTIER_COUNT, FRONTIER_FEATURES  # type: ignore


def epsilon_greedy(eps:float):
    def epsilon_greedy_wrapper(base_policy):
        def policy(input):
            if np.random.random() < eps:
                output = np.zeros(ACTION_SPACE)
                output[np.random.randint(0,ACTION_SPACE)] = 1
                return output
            else:
                return base_policy(input)
        return policy
    return epsilon_greedy_wrapper

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
        obs[keys[k]] = input[k:k+FRONTIER_FEATURES]
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


def epsilon_greedy_deterministic(eps:float):
    return epsilon_greedy(eps=eps)(deterministic_policy)
