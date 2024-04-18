from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from baseline import Policy

STATE, ACTION, REWARD, NEXT_STATE = 0, 1, 2, 3

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using ordinary importance
    # sampling (Hint: Sutton Book p. 109, every-visit implementation is fine)
    #####################

    C, Q = np.zeros((env_spec.nS,env_spec.nA)), initQ
    for t in trajs:
        w, g = 1, 0
        for step in t[::-1]:
            s_a = (step[STATE], step[ACTION])
            g = g * env_spec._gamma + step[REWARD]
            
            C[s_a] += 1
            Q[s_a] += w/C[s_a] * (g - Q[s_a])
            w *= pi.action_prob(*s_a)/bpi.action_prob(*s_a)
    
    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using weighted importance
    # sampling (Hint: Sutton Book p. 110, every-visit implementation is fine)
    #####################

    C, Q = np.zeros((env_spec.nS,env_spec.nA)), initQ
    for t in trajs:
        w, g = 1, 0
        for step in t[::-1]:
            g = g * env_spec._gamma + step[REWARD]
            s_a = (step[STATE], step[ACTION])
            
            C[s_a] += w
            if (w > 0 and C[s_a] > 0):
                Q[s_a] += w/C[s_a] * (g - Q[step[STATE], step[ACTION]])
            w *= pi.action_prob(*s_a)/bpi.action_prob(*s_a)

    return Q
