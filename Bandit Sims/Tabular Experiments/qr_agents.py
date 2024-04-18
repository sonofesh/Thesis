import numpy as np
from collections import defaultdict
from sympy import *
from typing import Iterable, Tuple

from env import Env, EnvSpec
from baseline import Policy, QPolicy

class QRPolicy(QPolicy):
    def __init__(self, env_spec:EnvSpec, epsilon, n) -> None:
        super().__init__(env_spec, epsilon)
        self.n = n
        self.name = 'QR (e:{}, n:{})'.format(epsilon, n)
        self.replay = {} #maps 
    
    def update(self, state, q_state, e):
        a = np.argmax(q_state.sum(axis = 1))
        self.maxA[state] = a 

class ExQRPolicy(QPolicy):
    def __init__(self, env_spec:EnvSpec, epsilon, n, init_c) -> None:
        super().__init__(env_spec, epsilon)
        self.n = n
        self.name = 'Ex-Bonus (e:{}, n:{}, c_value: {})'.format(epsilon, n, init_c)

        self.init_c = init_c
    
    def update(self, state, q_state, e):
        qa = q_state.sum(axis = 1)

        median = np.median(q_state, axis = 1)
        mid = int(self.n/2)
        var = ((q_state[:, -mid:] - median[:, None]) ** 2).mean(axis = 1)
        # decay of c doesn't seem ideal - possible improvement here
        c_value = self.init_c * np.sqrt(np.log(e + 1)/(e + 1))
            
        a = np.argmax(qa + np.sqrt(var) * c_value)
        self.maxA[state] = a 
    
class BiasedQRPolicy(QRPolicy):
    def __init__(self, env_spec:EnvSpec, epsilon, n, decay, norm, epoch) -> None:
        super().__init__(env_spec, epsilon, n)
        self.name = 'Biased (e:{}, n:{}, decay: {}, norm: {}, epoch: {})'.format(epsilon, n, decay, norm, epoch)

        #weights reduced by a factor of decay every quantile, weights are normalized to sum to one
        self.decay = decay
        self.w = get_weights(decay, n)

        self.norm = norm
        self.epoch = epoch
    
    def update(self, state, q_state, e):
        if (e + 1) % self.epoch == 0: self.decay_weights()
        qa = (q_state * self.w).sum(axis = 1)
        self.maxA[state] = np.argmax(qa)
    
    def decay_weights(self):
        # slowly normalize the weights back to one
        if self.norm:
            self.decay*= max((1 + self.norm), 1)
            self.w = get_weights(self.decay, self.n)
            self.count = 0

class BiasedQRPolicy_V2(QRPolicy): 
    def __init__(self, env_spec:EnvSpec, epsilon, n, decay, norm, trans, limit = 1) -> None:
        super().__init__(env_spec, epsilon, n)
        self.name = 'Biased_2 (e:{}, n:{}, decay: {}, norm: {}, trans: {})'.format(epsilon, n, decay, norm, trans)
        #weights calculated by the exp distribution, +decay indicates risk aversion -decay indicates risk preference
        self.decay = decay
        self.rate = decay
        self.limit = 1
        self.w = get_exp_weights(decay, n)

        #increase risk preference towards end of episode 
        self.norm_factor = norm
        self.step, self.current_episode, self.avg_length = 0, 0, 0
        self.l_alpha = .25
        self.trans, self.epoch = trans, np.inf

    def update(self, state, q_state, e):
        if e > self.current_episode: 
            #update rolling avg length estimate
            self.avg_length += self.l_alpha * (self.step - self.avg_length)
            if self.avg_length > self.trans: 
                #print(self.avg_length)
                self.epoch = round(self.avg_length/self.trans)
            else: print('short episode') 
            
            self.step = 0
            self.current_episode = e
            self.rate = self.decay

        if (self.step + 1) % 5 == 0:
            if abs(self.rate) < 1:
                self.rate += self.norm_factor
                self.w = get_exp_weights(self.rate, self.n)
        
        qa = (q_state * self.w).sum(axis = 1)
        self.maxA[state] = np.argmax(qa)
        self.step += 1

def get_weights(decay_factor, n):
    decay = abs(decay_factor)
    scale = (1. - decay)/(1. - pow(decay, n)) if decay != 1 else 1./n

    w = np.full(n, decay).cumprod() * 1./decay * scale
    #if positive weights flip to reflect higher importance for larger quantiles
    if decay_factor > 0: w = np.flip(w)
    #print(decay_factor, w)
    return w

def get_cpw_weights(q, B = .73):
    t = Symbol('t')
    y = pow(t, B)/pow((pow(t, B) + pow(1 - t, B)), 1/B)
    y_prime = y.diff(t)

    f = lambdify(t, y_prime, 'numpy')
    return f(q)/f(q).sum()

def get_exp_weights(rate, n):
    t = Symbol('t')
    if rate == 0: rate = .01
    y = rate * pow(np.e, -rate * t)

    f = lambdify(t, y, 'numpy')
    w = np.ones(n).cumsum()
    return f(w)/f(w).sum()

control_methods = ['sarsa', 'expected_sarsa', 'q_learning']

def qr_qlearning(
    env:Env,
    num_episodes:int,
    alpha:float,
    initQ:defaultdict,
    initPi:Policy,
    n: int,
    log = False,
    control_method = 'q_learning',
) -> Tuple[np.array,Policy, np.array]:
    """
    input:
        env: environment
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
        G: discounted return for first 
    """

    env_spec = env._env_spec
    Q, pi, G = initQ, initPi, []
    quantiles = np.full(n, 1./n).cumsum() - np.full(n, 1./(n * 2)) #initialize quantiles with 1/n gap
    assert control_method in control_methods, "Control method %s not supported".format(control_method)

    def qr_update(step, gamma):
        s, a, r, s_prime, a_prime = step
        max_action = pi.max_action(s_prime)
        x = Q[s][a]
        if control_method == 'sarsa': x_prime = Q[s_prime][a_prime]
        elif control_method == 'q_learning': x_prime = Q[s_prime][max_action]
        elif control_method == 'expected_sarsa': 
            p = pi.action_prob(s_prime)
            x_prime = np.sum(Q[s_prime] * p[:, np.newaxis], axis = 0) #expectation of π(a'|s') * q(s', a')

        for i, tau in enumerate(quantiles):
            indicator = 1 if (r + gamma * x_prime.mean()) < x[i] else 0
            x[i] = x[i] + alpha * (tau - indicator)
        
        return x
    
    for e in range(num_episodes):
        s, done, g = env.reset(), False, 0
        a = pi.epsilon_greedy(s)

        while (not done):
            s_prime, r, done = env.step(a)
            a_prime = pi.epsilon_greedy(s_prime)
            g = g * env_spec.gamma + r
            step = [s, a, r, s_prime, a_prime]
            Q[s][a] = qr_update(step, env_spec.gamma)
            #if log: print(s, a, Q[s])
            pi.update(s, Q[s], e)
            s, a = s_prime, a_prime
        
        if log: print("Return for episode ", e, g)
        G.append(g)
    
    return Q, pi, np.array(G)