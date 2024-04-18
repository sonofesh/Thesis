import numpy as np
from collections import defaultdict
from typing import Iterable, Tuple

from env import Env, EnvSpec

class Policy(object):
    def action_prob(self,state:int,action:int) -> float:
        """
        input: state, action
        return: \pi(a|s)
        """
        raise NotImplementedError()

    def action(self,state:int) -> int:
        """
        input: state
        return: action
        """
        raise NotImplementedError()

class QPolicy(Policy):
    def __init__(self, env_spec:EnvSpec, epsilon):
        self.nA = env_spec.nA
        self.maxA = {}
        self.epsilon = epsilon
    
    def action_prob(self,state:int) -> Iterable[float]:
        """
        input: state
        return: \pi(s)
        """
        if state in self.maxA:
            action_probs = np.full(self.nA, self.epsilon/self.nA)
            action_probs[self.max_action(state)] += (1 - self.epsilon)
        else:
            action_probs = np.full(self.nA, 1./self.nA)
        
        return action_probs
    
    def epsilon_greedy(self, state:int) -> int:
        """
        input: state, action
        return: epsilon greedy action
        """
        return np.random.choice(self.nA, 1, p = self.action_prob(state)).item()

    def max_action(self,state:int) -> int:
        """
        input: state
        return: action
        """
        if state in self.maxA: return self.maxA[state]
        return np.random.choice(self.nA)

    def update(self, state, action):
        self.maxA[state] = action

STATE, ACTION, REWARD, NEXT_STATE, A_PRIME = 0, 1, 2, 3, 4

BASELINES = ['sarsa', 'expected_sarsa', 'q_learning']

def one_step(
    env:Env,
    num_episodes:int,
    alpha:float,
    initQ:defaultdict,
    control_method = 'sarsa',
    elipson = .1,
    log = False,
) -> Tuple[np.array,Policy]:
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

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################

    env_spec = env._env_spec
    Q, pi, G = initQ, QPolicy(env_spec, elipson), []
    assert control_method in BASELINES, "Control method %s not supported".format(control_method)

    def td_error(step, gamma):
        s, a, r, s_prime, a_prime = step
        if control_method == 'sarsa': 
            return r + (gamma * Q[s_prime][a_prime]) - Q[s][a]

        elif control_method == 'expected_sarsa': 
            action_probs = pi.action_prob(s_prime)
            return r + (gamma * np.dot(action_probs, Q[s_prime])) - Q[s][a]
        
        elif control_method == 'q_learning':
            max_action = pi.max_action(s_prime)
            return r + (gamma * Q[s_prime][max_action]) - Q[s][a]
    
    for e in range(num_episodes):
        s, done, g = env.reset(), False, 0
        a = pi.epsilon_greedy(s)

        while (not done):
            s_prime, r, done = env.step(a)
            a_prime = pi.epsilon_greedy(s_prime)
            g = g * env_spec.gamma + r
            step = [s, a, r, s_prime, a_prime]

            Q[s][a] = Q[s][a] + alpha * td_error(step, env_spec.gamma)
            #if log: print(s, a, Q[s])
            pi.update(s, np.argmax(Q[s]))
            s, a = s_prime, a_prime
        
        if log: print("Return for episode ", e, g)
        G.append(g)
    
    return Q, pi, np.array(G)


def n_step_sarsa(
    env:Env,
    num_episodes:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
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

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################

    env_spec = env._env_spec
    Q, pi =  initQ, QPolicy(env_spec)

    return Q, pi

    for ep in len(num_episodes):
        s, done = env.reset()
        for t, step in enumerate(ep):
            g = g * env_spec._gamma + step[REWARD]
            s, a = step[STATE], step[ACTION]
            tau = t - n + 1

            if tau >= 0:
                if tau + n < len(ep): 
                    next_step = ep[t + 1]
                    g += pow(env_spec._gamma, n) * Q[next_step[STATE], next_step[ACTION]]
                
                s_tau, a_tau = ep[tau][STATE], ep[tau][ACTION]
                Q[s_tau, a_tau] += alpha * (g - Q[s_tau, a_tau])
                pi.update(s_tau, np.argmax(Q[s_tau]))
                
                p, g = 1, 0
            else: p *= pi.action_prob(s, a)/bpi.action_prob(s, a)

    
