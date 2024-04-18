import numpy as np
import pandas as pd

from typing import Iterable, Tuple, LiteralString

class Bandit():
    def __init__(self, k, bandit_seed = 0):
        self.k = k
        self.q_star = np.zeros(k)
        self.rg = np.random.RandomState(seed=bandit_seed)
    
    def reset(self, episode_seed = None): # based on seed, different episode will be generated
        if episode_seed is None:
            episode_seed = int(self.rg.randint(0, 100000000))

        self.episode_rg = np.random.RandomState(seed=episode_seed)
        self.q_star.fill(0)
    
    def best(self):  # returns optimal action
        return np.argmax(self.q_star)


class Distribution():
    def __init__(self, base_dist, scale, shift):
        self.base_dist = base_dist
        self.scale = scale
        self.shift = shift
    
    def sample(self):
        return (self.base_dist.rvs(1) + self.shift) * self.scale


class NonStationaryBandit(Bandit):
    def __init__(self, k = 10, bandit_seed = 0, drift = .01):
        super().__init__(k, bandit_seed)

    def step(self, a):
        reward = self.episode_rg.randn() + self.q_star[a]
        self.q_star += self.episode_rg.randn(self.k) * self.drift
        
        return reward


class StationaryBandit(Bandit):
    def __init__(self, arm_dist, sample_dist, scale = 1, k = 10, save = False):
        super().__init__(k)
        self.arm_dist = arm_dist
        self.sample_dist = sample_dist
        self.scale = scale
        self.save = save

        #init arm means and distributions
        self.q_star = arm_dist.rvs(k)
        self.q_arms = [Distribution(sample_dist, scale=scale, shift=u) for u in self.q_star]
    
    def reset(self, episode_seed = None): # based on seed, different episode will be generated
        self.q_star = self.arm_dist.rvs(self.k)
        self.q_arms = [Distribution(self.sample_dist, scale=self.scale, shift=u) for u in self.q_star]

    def step(self, a):
        reward = self.q_arms[a].sample()
        return reward
    
    def get_paramters(self):
        return {'mean arm': self.q_star, 'arm dist': self.q_arms}


def run_episode(bandit, algorithm, steps, episode_seed=None):
    bandit.reset(episode_seed)
    algorithm.reset()

    rs = []
    best_action_taken = []

    for __ in range(steps):
        a = algorithm.policy()
        best = bandit.best()
        r = bandit.step(a)
        algorithm.update(a, r)

        rs.append(r)
        best_action_taken.append(int(a == best))

    return np.array(rs), np.array(best_action_taken)

def experiment(bandit, agents, bandit_runs = 300, steps = 10000):
    outputs = {}
    if type(agents) != list: agents = [agents]
    for i, algo in enumerate(agents):
        average_rs = np.zeros(steps)
        average_best_action_taken = np.zeros(steps)

        #run multiple experiments
        for __ in range(bandit_runs):
            rs, best_action_taken = run_episode(bandit, algo, steps)
            average_rs += rs
            average_best_action_taken += best_action_taken
        
        print("Agent {}".format(algo.name))
        print(average_rs[-1]/bandit_runs, average_best_action_taken[-1]/bandit_runs)
        outputs[algo.name] = [average_rs/bandit_runs, average_best_action_taken/bandit_runs]

    #formulate to a pandas DataFrame
    return outputs


