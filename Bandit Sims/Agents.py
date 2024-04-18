import random
import numpy as np

class EpsilonGreedy():
    def __init__(self, dist, elipson, n_arms, init = 0):
        self.elipson = elipson
        self.n_arms = n_arms
        self.dist = dist #summarizes the distribution over n arms

        #init state variables
        self.means = [init] * n_arms
        self.counts = [0] * n_arms
    
    # returns arm/reward pair and runs update
    def pull_arm(self):
        arm = self._choose()
        reward = self.dist[arm].rvs[0]
        self._update(arm, reward)

        return (arm, reward)

    # returns arm pulled by policy
    def _choose(self) -> int:
        return np.argmax(self.means)
    
    # updates policy
    def _update(self, arm, reward):
        self.counts[arm] += 1
        self.means[arm] += (1/self.counts[arm]) * (reward - self.means[arm])
        
        
        

            
        