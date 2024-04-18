import random
import numpy as np

class Agent():
    def __init__(self, k, elipson, init_value):
        self.k = k
        self.elipson = elipson
        self.init_value = float(init_value)
        self.q = np.full(k, self.init_value)

    def reset(self):
        raise NotImplementedError

    def update(self, a, r):
        raise NotImplementedError

    def policy(self):
        if np.random.rand() < self.elipson: return np.random.randint(self.k)
        else: return np.argmax(self.q)


class SampleAverage(Agent):
    def __init__(self, k, elipson = 0, init_value = 0):
        super().__init__(k, elipson, init_value)
        self.counts = np.zeros(k) 
        self.name = 'SampleAverage'

    def reset(self):
        self.q.fill(self.init_value)
        self.counts.fill(0)

    def update(self, a, r):
        self.counts[a] += 1
        self.q[a] += (1/self.counts[a]) * (r - self.q[a])


class ConstantStepSize(Agent):
    def __init__(self, alpha, k, elipson = 0, init_value = 0):
        super().__init__(k, elipson, init_value)
        self.alpha = alpha
        self.name = 'ConstantStepSize'
        
    def reset(self):
        self.q.fill(self.init_value)

    def update(self, a, r):
        self.q[a] += self.alpha * (r - self.q[a])


class UCB(Agent):
    def __init__(self, c_value, k, init_value = 0):
        super().__init__(k, 0, init_value)
        self.c_value = c_value
        self.counts = np.zeros(k)
        self.steps = 0
        self.name = 'UCB'

    def reset(self):
        self.q.fill(self.init_value)
        self.counts.fill(0)
        self.steps = 0

    def update(self, a, r):
        self.counts[a] += 1
        self.q[a] += (1.0/self.counts[a]) * (r - self.q[a])
        self.steps += 1

    def policy(self):
        e = [self._calculate_estimate(q, c) for q, c in zip(self.q, self.counts)]
        return np.argmax(e)
    
    def _calculate_estimate(self, mean, count):
        if count == 0: return mean + self.c_value
        return mean + self.c_value * np.sqrt(np.log(self.steps)/count)
        

class Gradient(Agent):
    def __init__(self, alpha, k, init_value = 0):
        super().__init__(k, 0, init_value)
        self.alpha = alpha
        self.counts = np.zeros(k)
        self.h = np.zeros(k)
        self.name = 'Gradient'
    
    def reset(self):
        probs = self._calculate_prob()
        self.q.fill(self.init_value)
        self.counts.fill(0)
        self.steps = 0

    def update(self, a, r):
        #update prob
        probs = self._calculate_prob()
        for i in range(self.k):
            if i == a:
                self.h[i] += self.alpha * (r - self.q[i]) * (1 - probs[i])
            else:
                self.h[i] -= self.alpha * (r - self.q[i]) * (probs[i])

        #update rewards
        self.counts[a] += 1
        self.q[a] += (1/self.counts[a]) * (r - self.q[a])
    
    def policy(self):
        probs = self._calculate_prob()
        return np.random.choice(self.k, p=probs)

    def _calculate_prob(self):
        return np.exp(self.h)/(np.exp(self.h).sum())


    

            
        