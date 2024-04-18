import numpy as np

"""
Test traditional and distributional RL algorithms on the MountainCar Problem 
using tile coding 
"""

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width

        dim = len(tile_width)
        self.tiles = np.ceil((state_high - state_low)/tile_width).astype(int) + 1
        self.num_tiles = np.prod(self.tiles)

        tile_index = np.full((dim, num_tilings), range(num_tilings)).T
        self.offsets = (tile_index/num_tilings) * tile_width
        #self.tile_values = np.zeros(np.append(num_tilings, self.tiles))

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_actions * self.num_tilings * self.num_tiles
    
    def _tile_index(self, tile, num_tiling):
        """
        converts n-dim tile representation into scalar index
        """
        index = num_tiling * self.num_tiles
        for i, x in enumerate(tile):
            loc = np.prod(self.tiles[i+1:]) 
            index += x * loc

        return index

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        vector = np.zeros(self.feature_vector_len())
        rs = s - self.state_low

        if not done:
            for i, offset in enumerate(self.offsets):
                start_pos = rs + offset
                tile = np.floor(start_pos/self.tile_width).astype(int)
                
                action_index = a * self.num_tilings * self.num_tiles
                tile_index = self._tile_index(tile, i)
                vector[action_index + tile_index] = 1
        
        #print(vector)
        return vector


def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.01):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))
    
    for e in range(num_episode):
        s, done = env.reset(), False

        a = epsilon_greedy_policy(s, done, w)
        x = X(s, done, a)
        z = np.zeros((X.feature_vector_len()))
        q_old = 0

        while (not done):
            s_prime, r, done, _ = env.step(a)
            a_prime = epsilon_greedy_policy(s_prime, done, w)
            x_prime = X(s_prime, done, a_prime)

            q = np.dot(w, x)
            q_prime = np.dot(w, x_prime)
            td = r + gamma * q_prime - q
            
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w = w + alpha * (td + q - q_old) * z - alpha * (q - q_old) * x

            q_old = q_prime
            x = x_prime
            a = a_prime
        
        #if e % 100 == 0: print(e, " episodes completed")

    return w
