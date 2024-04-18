import random
import numpy as np
from Baseline import Agent
from collections import defaultdict

class QRAgent():
    def __init__(self, k, batch_update = False, batch_size = 10, epsilon = .05, n = 32, alpha = 0.1):
        self.k = k
        self.epsilon = epsilon
        self.q = np.zeros([k, n])

        self.n = n
        self.alpha = alpha
        self.t = np.full(n, 1./n).cumsum() - np.full(n, 1./(n * 2)) #initialize quantiles with 1/n gap
        
        self.batch_update = batch_update
        self.batch_size = batch_size
        self.d = defaultdict(list)

        self.name = 'QRAgent (e:{}, n:{}, alpha:{}, batch_update:{})'.format(
                                    epsilon, n, alpha, batch_update)

    def reset(self):
        self.q = np.zeros([self.k, self.n])

    def update(self, a, r):
        if self.batch_update:
            self.d[a].append(r)

            #select b random elements from past data if batch is too big
            batch = self.d[a] 
            if len(batch) > self.batch_size: batch = np.random.choice(batch, self.batch_size)
            for sr in batch: self._update_q(a, sr)
            
        else: self._update_q(a, r)
        
    def _update_q(self, a, r):
        for i, tau in enumerate(self.t):
            indicator = 1 if r < self.q[a][i] else 0
            self.q[a][i] += self.alpha * (tau - indicator)

    def policy(self):
        if np.random.rand() < self.epsilon: return np.random.randint(self.k)
        else: return np.argmax(self.q.sum(axis = 1))
    
    def get_weights(self):
        return self.q
    

class BiasedQRAgent(QRAgent):
    def __init__(self, k, decay_factor, norm_factor = 0, softmax = False, 
                 batch_update = False, batch_size = 10, 
                 epsilon = .05, n = 32, alpha = 0.05):
        
        super().__init__(k, batch_update, batch_size, epsilon, n, alpha)
        self.softmax = softmax
        
        #weights reduced by a factor of decay every quantile, weights are normalized to sum to one
        self.decay_factor = decay_factor
        self.w = update_weights(decay_factor, n)

        self.norm_factor = norm_factor
        self.count = 0
        self.epoch = 500

        self.name = 'Biased QRAgent (e:{}, n:{}, decay: {}, norm_factor: {}, alpha:{})'.format(
                                    epsilon, n, decay_factor, norm_factor, alpha)

    def policy(self):
        qa = (self.q * self.w).sum(axis = 1)
        # slowly normalize the weights back to one
        if self.norm_factor and abs(self.decay_factor) < 1:
            self.count += 1
            if self.count >= self.epoch:
                self.decay_factor *= (1 + self.norm_factor)
                self.w = update_weights(self.decay_factor, self.n)
                self.count = 0

        if self.softmax:
            qa = softmax(qa * 10)
            #if np.random.rand() < .05: print(qa)
            return np.random.choice(self.k, 1, p=list(qa)).item()
        else:
            if np.random.rand() < self.epsilon: return np.random.randint(self.k)
            else: return np.argmax(qa)


class MultiQRAgent(QRAgent):
    def __init__(self, k, norm_factor = 0, estimators = 11, 
                 batch_update = False, batch_size = 10, 
                 epsilon = .05, n = 32, alpha = 0.05):
        
        super().__init__(k, batch_update, batch_size, epsilon, n, alpha)
        self.init_decay = np.linspace(0, 1, estimators)
        self.agents = [BiasedQRAgent(k, decay_factor=decay, norm_factor=norm_factor)
                       for decay in self.init_decay]
        self.index = 0

        self.name = 'Multi QRAgent (e:{}, n:{}, estimators: {}, norm_factor: {}, alpha:{})'.format(
                                    epsilon, n, estimators, norm_factor, alpha)

    def policy(self):
        agent = self.agents[self.index % len(self.agents)]
        self.index += 1

        return agent.policy()
    
class BonusQRAgent(QRAgent):
    def __init__(self, k, init_c, batch_update = False, batch_size = 10, 
                 epsilon = .05, n = 32, alpha = 0.05):
        
        super().__init__(k, batch_update, batch_size, epsilon, n, alpha)
        self.init_c = init_c
        self.counts = np.ones(k)
        print(init_c)

        self.name = 'Exploration Bonus QRAgent (e:{}, n:{}, c_value: {}, alpha:{}, batch_update:{})'.format(
                                                epsilon, n, init_c, alpha, batch_update)

    def policy(self):
        qa = self.q.sum(axis = 1)
        #self.steps += 1

        if np.random.rand() < self.epsilon: return np.random.randint(self.k)
        else:
            median = np.median(self.q, axis = 1)
            mid = int(self.n/2)
            var = ((self.q[:, -mid:] - median[:, None]) ** 2).mean(axis = 1)
            # decay of c doesn't seem ideal - possible improvement here
            c_value = self.init_c * np.sqrt(np.log(self.counts)/self.counts)
            
            return np.argmax(qa + np.sqrt(var) * c_value)
    
    def update(self, a, r):
        super().update(a, r)
        self.counts[a] += 1


### Helper Methods ###
def softmax(q):
    return np.exp(q)/np.sum(np.exp(q), axis=0)

def update_weights(decay_factor, n):
    decay = abs(decay_factor)
    scale = (1. - decay)/(1. - pow(decay, n)) if decay != 1 else 1./n

    w = np.full(n, decay).cumprod() * 1./decay * scale
    if decay_factor > 0: w = np.flip(w)
    return w
    


