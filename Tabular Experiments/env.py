import numpy as np
from typing import Tuple, List, Iterable

class EnvSpec(object):
    def __init__(self,nS,nA,gamma,bounds=()):
        self._nS = nS
        self._nA = nA
        self._gamma = gamma

        if len(bounds) > 0: 
            self._dim = len(bounds)
            self._bounds = bounds
        else: 
            self._dim = 1
            self._bounds = [(nS)]

    @property
    def nS(self) -> int:
        """ # possible states """
        return self._nS
    
    @property
    def dim(self) -> int:
        """ # possible states """
        return self._dim

    @property
    def bounds(self) -> Iterable[List[int]]:
        """ return all bounds """
        return self._bounds
    
    @property
    def nA(self) -> int:
        """ # possible actions """
        return self._nA

    @property
    def gamma(self) -> float:
        """ discounting factor of the environment """
        return self._gamma

    def get_bound(self, axis = 0) -> List[int]:
        """ bounds for specific axis """
        assert axis < self.dim, "Axis out of bounds"
        return self.bounds[axis]
    

class Env(object):
    def __init__(self,env_spec):
        self._env_spec = env_spec

    @property
    def spec(self) -> EnvSpec:
        return self._env_spec

    def reset(self) -> int:
        """
        reset the environment. It should be called when you want to generate a new episode
        return:
            initial state
        """
        raise NotImplementedError()

    def step(self,action:int) -> (int, int, bool):
        """
        proceed one step.
        return:
            next state, reward, done (whether it reached to a terminal state)
        """
        raise NotImplementedError()

class EnvWithModel(Env):
    @property
    def TD(self) -> np.array:
        """
        Transition Dynamics
        return: a numpy array shape of [nS,nA,nS]
            TD[s,a,s'] := the probability it will resulted in s' when it execute action a given state s
        """
        raise NotImplementedError()

    @property
    def R(self) -> np.array:
        """
        Reward function
        return: a numpy array shape of [nS,nA,nS]
            R[s,a,s'] := reward the agent will get it experiences (s,a,s') transition.
        """
        raise NotImplementedError()

