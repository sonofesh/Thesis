import threading
from multiprocessing import Process, Event, Pipe
from multiprocessing.managers import BaseManager
from sympy import *
from typing import Iterable, Callable
import copy
import gym
import time

import torch.nn as nn
import numpy as np
import torch

from rl_utils import ParallelReplayMemory, TPReplayMemory
from dqn_qr import QR_Network, evaluate

import warnings
warnings.filterwarnings("ignore")

class Biased_QR_Network(QR_Network):
    def __init__(self, state_dims, num_actions, n, alpha, decay):
        super().__init__(state_dims, num_actions, n, alpha)
        self.set_decay(decay=decay)
    
    def max_action(self, states, epsilon=0):
        action_values = (self.forward(states) * self.w).sum(axis = 2)
        
        if np.random.random() >= epsilon: return torch.max(action_values, 1)[1]
        else: return torch.randint(high=self.num_actions, size=(len(states),))
    
    def set_decay(self, decay):
        self.decay = decay
        self.w = torch.Tensor(get_exp_weights(decay, self.n))

def get_exp_weights(rate, n):
    t = Symbol('t')
    if rate == 0: return np.full(n, 1./n)
    y = rate * pow(np.e, -rate * t)

    f = lambdify(t, y, 'numpy')
    w = np.ones(n).cumsum()
    return f(w)/f(w).sum()

class ActorProcess(Process):
    def __init__(self, env, qr_process_cp, results, memory, gamma, epsilon, t_id, stop, decay):
        Process.__init__(self)

        #copied objects
        self.env = copy.deepcopy(env)
        self.qr_process_cp = qr_process_cp
        qr = qr_process_cp.pop(t_id)
        self.qr = copy.deepcopy(qr)
        self.qr.set_decay(decay)

        #shared objects
        self.results = results
        self.memory = memory
        
        #primitives
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        
        #concurrency objects
        self.t_id = t_id
        self.stop = stop
    
    def run(self):
        G_0, step, first_success = [], 0, True
        print(self.memory)

        while (self.stop.is_set() == False):
            s, done, trunc, g = self.env.reset()[0], False, False, 0.
            while not (done | trunc):
                #build training batch
                a = self.qr.max_action(torch.tensor([s]), epsilon=self.epsilon).item()
                s_prime, r, done, trunc, _ = self.env.step(a)
                self.memory.push(self.t_id, step = (s, a, s_prime, r, float(done)))
                #print(self.t_id, self.memory)
                
                g = g * self.gamma + r
                s = s_prime
                step += 1

            if self.t_id in self.qr_process_cp:
                updated_qr = self.qr_process_cp.pop(self.t_id)
                qr.load_state_dict(updated_qr.state_dict())
                qr.set_decay(self.decay)
                G_0 = []

            G_0.append(g)
            step += 1
            
            self.results[self.t_id] = (len(G_0), round(np.mean(G_0), 2))
            if type(self.memory) == TPReplayMemory:  self.memory.update(self.t_id, self.results[self.t_id][1])
        
class CustomManager(BaseManager): pass

def gather_data(shared_memory, env, qr, gamma, epsilon, t_id, pipe, stop, decay):
    qr.set_decay(decay)
    G_0, step, first_success = [], 0, True

    while (stop.is_set() == False):
        s, done, trunc, g = env.reset()[0], False, False, 0.

        while not (done | trunc):
            #build training batch
            a = qr.max_action(torch.tensor([s]), epsilon=epsilon).item()
            s_prime, r, done, trunc, _ = env.step(a)
            shared_memory.push(t_id, step = (s, a, s_prime, r, float(done)))
            
            g = g * gamma + r
            s = s_prime
            step += 1
        
        if g > -200: print(t_id, g)
        if pipe.poll():
            state_dict = pipe.recv()
            qr.load_state_dict(state_dict)
            qr.set_decay(decay)
            G_0 = []

        G_0.append(g)
        if type(shared_memory) == TPReplayMemory:  shared_memory.update(t_id, round(np.mean(G_0), 2))

def process_data(env, qr, shared_memory, stop_signal, filename, connections, iterations, batch_size, refresh_rate, gamma):
    qr_target = copy.deepcopy(qr)

    i = 0
    while i < iterations:
        #refresh target network and evaluate returns
        if i % refresh_rate == 1: 
            state_dict = qr.state_dict()
            qr_target.load_state_dict(state_dict)
            for pipe in connections: pipe.send(state_dict)

        #process batch and update network
        if shared_memory.size() < batch_size: continue
        batch = shared_memory.sample(batch_size)
        states, actions, rewards, next_states, dones = batch
        z_theta = qr_target(next_states)[np.arange(len(states)), qr_target.max_action(next_states)]
        qr.update(batch, z_theta.detach(), gamma)
        i += 1
    
    #summary = evaluate(env, qr, gamma, 100)
    #print(summary.describe())
    torch.save(qr.state_dict(), filename)
    stop_signal.set()

def stop_processes(processes):
    for t_id in processes:
        t_id, stop, decay, process = t_id
        stop.set()
        process.join()

def train_parallel_qr(
    env, #open-ai environment
    qr:QR_Network,
    filename:str,
    memory_type:ParallelReplayMemory,
    memory_size:int,
    gamma:float,
    iterations:int,
    batch_size:int,
    epsilon:Callable,
    num_processes = 4,
    refresh_rate = 100,
    decay_rates = [],
    ) -> Iterable[float]:

    """ parallelize the data collection aspect of QR-DQN

    input:
        env: target environment; openai gym
        qr: network with desired hyperparameters
        target_qr: copy of qr network
        gamma: discount factor
        num_episode: # of training iterations
        batch_size: batch size
        eps: function to calculate epsilon from step size
        process: number of worker processes
        memory: database to draw samples from
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    if len(decay_rates) == 0: decay_rates = np.zeros(num_processes)
    
    CustomManager.register('SharedMemory', memory_type)
    CustomManager.register('Dict', dict)
    CustomManager.register('Network', QR_Network)

    with CustomManager() as manager:
        shared_memory = manager.SharedMemory(memory_size, num_processes)
        results = manager.Dict()
        processes = []

        #init actor processes
        args = [shared_memory, env, qr, gamma, epsilon]
        stop_signal = Event()
        connections = []
        for i in range(num_processes):
            #shared_memory, env, qr, gamma, epsilon, t_id, pipe, stop, decay
            parent, child = Pipe()
            connections.append(parent)
            t_info = [i, child, stop_signal, decay_rates[i]]
            temp_args = tuple(args + t_info)
                
            p = Process(target=gather_data, args=temp_args) #threading.Thread(target=gather_data, args=tuple(temp_args))
            processes.append(p)
            p.start()
            
        #init learner process
        if not stop_signal.is_set():
            #qr, shared_memory, stop_signal, filename, connections, iterations, batch_size, refresh_rate, gamma
            args = [env, qr, shared_memory, stop_signal, filename, connections, iterations, batch_size, refresh_rate, gamma]
            p = Process(target=process_data, args=args)
            processes.append(p)
            p.start()
            
        #stop ongoing threads
        for p in processes: p.join()
    
    qr.load_state_dict(torch.load(filename))
    qr.eval()
    return qr

if __name__ == "__main__":
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    N = 4
    alpha = 1e-3
    num_processes = 5
    eps_start, eps_end, eps_dec = 0.9, 0.05, 5000
    eps = lambda steps: eps_end + (eps_start - eps_end) * np.exp(-1. * steps / eps_dec)
    eps = .1

    """
    env, #open-ai environment
    qr:QR_Network,
    filename:String,
    memory_type:ParallelReplayMemory,
    memory_size:int,
    gamma:float,
    iterations:int,
    batch_size:int,
    epsilon:Callable,
    num_processes = 4,
    refresh_rate = 100,
    decay_rates = [],
    """
    params = {'state_dims': env.observation_space.shape[0], 'num_actions': env.action_space.n, 'n': N, 'alpha': alpha, 'decay': 0}
    training_params = {'env': env, 'gamma': 1., 'iterations':20000, 'batch_size':128, 'epsilon':eps, 'num_processes':num_processes, 'refresh_rate':50}

    #test biased DDQN-QR
    qr = Biased_QR_Network(**params)
    p_replay = ParallelReplayMemory(20000, num_processes)
    tp_replay = TPReplayMemory(20000, num_processes)
    
    #tp_replay = ParallelReplayMemory(10000, num_processes)
    training_params['filename'] = 'model_scripted.pt'
    training_params['decay_rates'] = np.array([-.2, -.1, .0, .1, .2])
    training_params['qr'] = qr
    training_params['memory_type'] = ParallelReplayMemory
    training_params['memory_size'] = 20000
    
    G = train_parallel_qr(**training_params)
    G_test = evaluate(env, qr, gamma=1, num_episodes=100)
    print(G_test.describe())

    #test unbiased DDQN-QR
    qr = Biased_QR_Network(**params)
    training_params['decay_rates'] = []
    training_params['qr'] = qr
    training_params['memory_type'] = ParallelReplayMemory
    training_params['memory_size'] = 20000
    
    G = train_parallel_qr(**training_params)
    G_test = evaluate(env, qr, gamma=1, num_episodes=100)
    print(G_test.describe())
    
    