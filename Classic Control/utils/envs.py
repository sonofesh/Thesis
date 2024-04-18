import gym
import torch
import copy
import numpy as np
from sympy import *

class RiskFunction():
    def __init__(self, init, function, n, period, update_freq=1):
        self.n = n
        self.period = period
        self.function = lambda x: function((x/period) + init)
        
        self.rate = init
        self.w = get_exp_weights(self.rate, n)
        self.update_freq = update_freq
        
    def get_weights(self, step):
        if step % self.update_freq == 0:
            self.rate = self.function(step)
            self.w = torch.Tensor(get_exp_weights(self.rate, self.n))

        return self.w
    
def get_exp_weights(rate, n):
    t = Symbol('t')
    if rate == 0: return np.full(n, 1./n)
    y = rate * pow(np.e, -rate * t)

    f = lambdify(t, y, 'numpy')
    w = np.ones(n).cumsum()
    return f(w)/f(w).sum()

    
class MultiStep():
    def __init__(self, env, qr, actors, risk_functions, memory, eps):
        #init environments
        self.states, self.envs = [], []
        for i in range(actors):
            temp = copy.deepcopy(env)
            self.states.append(temp.reset(seed = i)[0])
            self.envs.append(temp)
        
        self.eps = eps
        self.qr = qr
        self.actors = actors
        self.risk_functions = risk_functions
        self.memory = memory
        self.steps = [0] * actors
        self.count = 0

    def gen(self):
        epsilon = self.eps(self.count)
        for i in range(self.actors):
            risk_func, s, env = self.risk_functions[i], self.states[i], self.envs[i]
            
            w = risk_func.get_weights(self.steps[i])
            #a = self.qr.max_biased_action(torch.tensor([s]), w = w, epsilon = epsilon).item()
            a = self.qr.max_action(torch.tensor([s]), epsilon=epsilon).item()
            s_prime, r, done, trunc, _ = env.step(a)
            self.memory.push(s, a, s_prime, r, float(done))
            
            self.steps[i] += 1
            self.states[i] = s_prime

            if done or trunc: 
                self.states[i] = env.reset()[0]
                self.steps[i] = 0
        
        self.count += 1


