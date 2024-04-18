import torch 
import random
import numpy as np
from utils.sum_tree import SumSegmentTree, MinSegmentTree

# Helper Methods
def softmax(q):
    return np.exp(q)/np.sum(np.exp(q), axis=0)

# Memory Classes 
class ReplayMemory:
    def __init__(self, capacity):
        """Create Prioritized Replay buffer.

        capacity: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are written over
        """

        self.capacity = capacity
        self.memory = []
        self.new_memory = []
        self.next_indx = 0
        self.type = 'regular'

    def push(self, state, action, next_state, reward, done):
        transition = torch.Tensor([state]), torch.Tensor([action]), torch.Tensor([next_state]), torch.Tensor([reward]), torch.Tensor([done])
        
        if self.next_indx >= len(self.memory): self.memory.append(transition)
        else: self.memory[self.next_indx] = transition
        self.next_indx = (self.next_indx + 1) % self.capacity

        self.new_memory.append(transition)

    def _construct_sample(self, index, additional_samples=[]):
        sample = additional_samples
        for i in index: sample.append(self.memory[i])
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*sample)

        batch_state = torch.cat(batch_state)
        batch_action = torch.cat(batch_action).int()
        batch_reward = torch.cat(batch_reward)
        batch_done = torch.cat(batch_done)
        batch_next_state = torch.cat(batch_next_state)

        return batch_state, batch_action, batch_reward.unsqueeze(1), batch_next_state, batch_done.unsqueeze(1)
        
    def sample(self, batch_size):
        new_samples = self.new_memory.copy()
        self.new_memory = []

        sample_index = random.sample(range(len(self.memory)), batch_size - len(new_samples))
        return self._construct_sample(sample_index, additional_samples=new_samples)
    
    def __len__(self):
        return len(self.memory)


class PriorityReplayMemory(ReplayMemory):
    def __init__(self, capacity, alpha):
        """Create Prioritized Replay buffer.

        Parameters:
        capacity: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are written over
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        """

        super().__init__(capacity)
        assert alpha >= 0
        self.alpha = alpha

        #capacity rounded up to nearest power of 2
        it_capacity = pow(2, np.ceil(np.log2(capacity))) 
        self._it_sum = SumSegmentTree(int(it_capacity))
        self._it_min = MinSegmentTree(int(it_capacity))
        self.max_priority = 1
        self.type = 'priority'

    def push(self, state, action, next_state, reward, done):
        indx = self.next_indx
        super().push(state, action, next_state, reward, done)

        self._it_sum[indx] = self.max_priority ** self.alpha
        self._it_min[indx] = self.max_priority ** self.alpha

    def _sample_proportional(self, batch_size):
        results = []
        p_total = self._it_sum.sum(0, len(self.memory) - 1)
        every_range_len = p_total / batch_size
        
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            indx = self._it_sum.find_prefixsum_idx(mass)
            results.append(indx)
        
        return results

    def sample(self, batch_size, beta):
        """Sample a batch of experiences. Also returns importance weights and idxes
        of sampled experiences.

        beta: degree to use importance weights (0 - no corrections to full correction)
        """

        assert beta > 0 
        sample_index = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.memory)) ** (-beta)

        for i in sample_index:
            p_sample = self._it_sum[i] / self._it_sum.sum()
            weight = (p_sample * len(self.memory)) ** (-beta)
            weights.append(weight / max_weight)

        weights = torch.Tensor(weights)
        batch = self._construct_sample(sample_index)

        return batch, weights, sample_index

    def update_priorities(self, index, priorities):
        """Update priorities of sampled transitions. Sets priority of transition at 
        index idxes[i] in buffer to priorities[i].
        """

        #print(len(index), priorities)
        assert len(index) == len(priorities)

        for i, priority in zip(index, priorities):
            assert priority >= 0
            assert 0 <= i < len(self.memory)

            self._it_sum[i] = priority ** self.alpha
            self._it_min[i] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)


class ParallelReplayMemory:
    def __init__(self, capacity, threads):
        self.capacity = capacity
        self.thread_capacity = capacity // threads
        self.memory = {}
        self.threads = threads

        for thread in range(threads): self.memory[thread] = []

    def push(self, thread, step):
        transition = tuple([torch.Tensor([x]) for x in step])
        self.memory[thread].append(transition)
        if len(self.memory[thread]) > self.thread_capacity: del self.memory[thread][0]

    def sample(self, batch_size):
        sample = [] 
        for thread in range(self.threads): 
            size = min(batch_size // self.threads, len(self.memory[thread]))
            sample += random.sample(self.memory[thread], size)
        return self.batch_sample(sample)
    
    def batch_sample(self, sample):
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*sample)
        
        batch_state = torch.cat(batch_state)
        batch_action = torch.cat(batch_action).int()
        batch_reward = torch.cat(batch_reward)
        batch_done = torch.cat(batch_done)
        batch_next_state = torch.cat(batch_next_state)

        return batch_state, batch_action, batch_reward.unsqueeze(1), batch_next_state, batch_done.unsqueeze(1)
    
    def size(self):
        return sum(len(s) for s in self.memory.values())

    def __len__(self):
        #print([len(s) for s in self.memory.values()])
        return sum(len(s) for s in self.memory.values())
    
class TPReplayMemory(ParallelReplayMemory):
    """
    Replay memory with thread-based priority. Sample higher performing threads with higher probability
    Use a simple constant step size methods to 
    """
    def __init__(self, capacity, threads, alpha=.1):
        super().__init__(capacity, threads)
        self.q = np.zeros(self.threads)
        self.alpha = alpha
    
    def update(self, thread, r):
        self.q[thread] += self.alpha * (r - self.q[thread])
    
    def sample(self, batch_size):
        sample_sizes = np.round(softmax(self.q) * batch_size).astype(int)
        if np.unique(sample_sizes).size > 1: print('sample diff')

        sample = [] 
        for i, sz in enumerate(sample_sizes): sample += random.sample(self.memory[i], max(sz, 1))
        return self.batch_sample(sample)
    
#ReplayMemory(10000)
#PrioritydReplayMemory(10000, alpha=.5)
