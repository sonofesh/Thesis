from env import EnvSpec, Env, EnvWithModel
import numpy as np 
import gym

class OneStateMDP(Env): # MDP introduced at Fig 5.4 in Sutton Book
        def __init__(self):
            env_spec=EnvSpec(2,2,1.)

            super().__init__(env_spec)
            self.final_state = 1
            self.trans_mat, self.r_mat = self._build_trans_mat()

        def _build_trans_mat(self):
            trans_mat = np.zeros((2,2,2))

            trans_mat[0,0,0] = 0.9
            trans_mat[0,0,1] = 0.1
            trans_mat[0,1,0] = 0.
            trans_mat[0,1,1] = 1.0
            trans_mat[1,:,1] = 1.

            r_mat = np.zeros((2,2,2))
            r_mat[0,0,1] = 1.

            return trans_mat, r_mat

        def reset(self):
            self._state = 0
            return self._state

        def step(self, action):
            assert action in list(range(self.spec.nA)), "Invalid Action"
            assert self._state != self.final_state, "Episode has ended!"

            prev_state = self._state
            self._state = np.random.choice(self.spec.nS,p=self.trans_mat[self._state,action])
            r = self.r_mat[prev_state,action,self._state]

            if self._state == self.final_state:
                return self._state, r, True
            else:
                return self._state, r, False

class CliffWalking(Env): # MDP introduced at Fig 5.4 in Sutton Book
        def __init__(self):
            bounds = np.array([(0, 11), (0, 2)])
            env_spec = EnvSpec(48,4,1., bounds)

            super().__init__(env_spec)
            self.start_state = (0, 0)
            self.final_state = (11, 0)

        def in_bounds(self, state):
            x_bound, y_bound = self._env_spec.get_bound(axis = 0), self._env_spec.get_bound(axis = 1)
            x, y = state

            if x < x_bound[0] or x > x_bound[1]: return False
            if y < y_bound[0] or y > y_bound[1]: return False
            return True

        def off_cliff(self, state):
            x, y = state
            x_bound = self._env_spec.get_bound(axis = 0)

            if (x_bound[0] < x < x_bound[1]) and (y == 0): return True
            return False

        def next_state(self, state, action):
            l, r, u, d = range(0, 4)
            next_state, reward = list(state), -1

            if action == l: next_state[0] -= 1
            if action == r: next_state[0] += 1
            if action == d: next_state[1] -= 1
            if action == u: next_state[1] += 1

            if not self.in_bounds(next_state): next_state = state #return to past state if invaild action
            if self.off_cliff(next_state):
                next_state = self.start_state
                reward = -100
            
            #print(state, action, next_state, reward)
            return tuple(next_state), reward

        def reset(self):
            self._state = self.start_state
            return self._state

        def step(self, action):
            assert action in list(range(self.spec.nA)), "Invalid Action"
            assert self._state != self.final_state, "Episode has ended!"

            prev_state = self._state
            self._state, r = self.next_state(prev_state, action)

            if self._state == self.final_state: return self._state, r, True
            else: return self._state, r, False

class TaxiDriver(Env):
    def __init__(self):
        self.env = gym.make("Taxi-v3")
        env_spec = EnvSpec(nS = self.env.observation_space.n, nA = self.env.action_space.n, gamma = 1.)
        super().__init__(env_spec)
    
    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        assert action in list(range(self.spec.nA)), "Invalid Action"

        s_prime, r, done, trunc, _ = self.env.step(0)
        return s_prime, r, (done | trunc)

class Maze(Env): # MDP introduced at Fig 5.4 in Sutton Book
        def __init__(self):
            bounds = np.array([(0, 11), (0, 2)])
            env_spec = EnvSpec(48,4,1., bounds)

            super().__init__(env_spec)
            self.start_state = (0, 0)
            self.final_state = (11, 0)

        def in_bounds(self, state):
            x_bound, y_bound = self._env_spec.get_bound(axis = 0), self._env_spec.get_bound(axis = 1)
            x, y = state

            if x < x_bound[0] or x > x_bound[1]: return False
            if y < y_bound[0] or y > y_bound[1]: return False
            return True

        def off_cliff(self, state):
            x, y = state
            x_bound = self._env_spec.get_bound(axis = 0)

            if (x_bound[0] < x < x_bound[1]) and (y == 0): return True
            return False

        def next_state(self, state, action):
            l, r, u, d = range(0, 4)
            next_state, reward = list(state), -1

            if action == l: next_state[0] -= 1
            if action == r: next_state[0] += 1
            if action == d: next_state[1] -= 1
            if action == u: next_state[1] += 1

            if not self.in_bounds(next_state): next_state = state #return to past state if invaild action
            if self.off_cliff(next_state):
                next_state = self.start_state
                reward = -100
            
            #print(state, action, next_state, reward)
            return tuple(next_state), reward

        def reset(self):
            self._state = self.start_state
            return self._state

        def step(self, action):
            assert action in list(range(self.spec.nA)), "Invalid Action"
            assert self._state != self.final_state, "Episode has ended!"

            prev_state = self._state
            self._state, r = self.next_state(prev_state, action)

            if self._state == self.final_state: return self._state, r, True
            else: return self._state, r, False