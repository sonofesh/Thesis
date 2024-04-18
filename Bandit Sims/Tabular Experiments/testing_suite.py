from collections import defaultdict
from functools import partial

import numpy as np
from tqdm import tqdm
import pandas as pd

from mdp import OneStateMDP, CliffWalking, TaxiDriver
from baseline import Policy, QPolicy, one_step, BASELINES as baselines
from qr_agents import QRPolicy, BiasedQRPolicy, ExQRPolicy, BiasedQRPolicy_V2, qr_qlearning

class RandomPolicy(Policy):
    def __init__(self,nA,p=None):
        self.p = p if p is not None else np.array([1/nA]*nA)

    def action_prob(self,state,action=None):
        return self.p[action]

    def action(self,state):
        return np.random.choice(len(self.p), p=self.p)

def one_state_val(episodes = 1000):
    # Sarsa, Q-Learning, and Expected Sarsa Sanity Check
    env = OneStateMDP()

    for method in baselines:
        initQ = defaultdict(partial(np.zeros, env.spec.nA))
        Q, pi, G = one_step(env, episodes, alpha=0.005, initQ=initQ, control_method=method)
        assert pi.max_action(0) == 0, "Policy failed to converge"

        # output results
        print('Results for ', method)
        print(Q)
        print('Average Reward: ', G.mean())

def test_baseline(env, episodes, alpha = .5, elipson = .1):
    env = CliffWalking()

    for method in baselines:
        initQ = defaultdict(partial(np.zeros, env.spec.nA))
        Q, pi, G = one_step(env, episodes, alpha = 0.5, elipson = elipson, initQ = initQ, control_method = method)
        #assert pi.max_action(0) == 0, "Policy failed to converge"

        # output results
        print('Results for ', method)
        print('Average Reward: ', G.mean())
        #print('Ending Rewards: ', G[-20:])
        

def test_qr(env, n, agent:Policy, episodes, control_method, alpha = .5):

    initQ = defaultdict(partial(np.zeros, (env.spec.nA, n)))

    Q, pi, G = qr_qlearning(env, episodes, alpha = alpha, initQ = initQ, initPi = agent, 
                            n = n, control_method = control_method)

    # output results
    print('Results for', agent.name, control_method)
    print('Average Reward: ', G.mean())
    return G

if __name__ == "__main__":
    one_state_val()
    
    # Experiment on Cliff Gridworld
    env = CliffWalking()
    n_episodes = 1000
    elipson = .1 
    test_baseline(env, episodes = n_episodes, alpha = .5, elipson = elipson)
    
    # Test Quantile Regression
    n = 8
    alpha = .5
    control_method = 'q_learning'
    params = {'env':env, 'n':n, 'episodes':n_episodes, 'control_method':control_method, 'alpha':.5}

    qr_policy = QRPolicy(env._env_spec, epsilon=elipson, n=n)
    params['agent'] = qr_policy
    test_qr(**params)

    ex_policy = ExQRPolicy(env._env_spec, epsilon=elipson, n=n, init_c = .5)
    params['agent'] = ex_policy
    test_qr(**params)

    #bias_policy = BiasedQRPolicy(env._env_spec, n=n, epsilon=elipson, decay=.5, norm=.9, epoch = 50)
    #params['agent'] = bias_policy
    #cliff_gridworld(**params)
    
    bias_policy_v2 = BiasedQRPolicy_V2(env._env_spec, n=n, epsilon=elipson, decay=-.25, norm=.05, trans=5)
    params['agent'] = bias_policy_v2
    test_qr(**params)
    
    policy_param = bias_policy_v2.__dict__
    policy_param.pop('maxA')
    policy_param.pop('name')
    print(policy_param)
    
    # Experiment on Maze Problem

    # Taxi Driver
    env = TaxiDriver()
    n_episodes = 1000
    elipson = .1
    test_baseline(env, episodes = n_episodes, alpha = .25, elipson = elipson)

    # Test Quantile Regression
    n = 8
    alpha = .75
    control_method = 'q_learning'
    params = {'env':env, 'n':n, 'episodes':n_episodes, 'control_method':control_method, 'alpha':alpha}

    qr_policy = QRPolicy(env._env_spec, epsilon=elipson, n=n)
    params['agent'] = qr_policy
    G = test_qr(**params)
    print(G[-50:])

    bias_policy_v2 = BiasedQRPolicy_V2(env._env_spec, n=n, epsilon=elipson, decay=-.25, norm=.05, trans=10)
    params['agent'] = bias_policy_v2
    G = test_qr(**params)
    print(G[-50:])
    
    # Ice Walker
    


    
    

