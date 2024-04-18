from typing import Iterable, Callable
from sympy import *
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import pandas as pd
import torch

from utils.memory import ReplayMemory, PriorityReplayMemory
from utils.envs import *
from models.qr_agents import QrNetwork, BiasedQr, RandomBiasQr, soft_update

device = torch.device("mps")
print(device)

def train_qr(
    env, #open-ai environment
    qr:QrNetwork,
    qr_target:QrNetwork,
    gamma:float,
    num_episodes:int,
    batch_size:int,
    eval_every:int,
    eps:Callable,
    memory = ReplayMemory(10000),
    refresh_rate = 100,
    ) -> Iterable[float]:
    """
    implement simple verison of QR-DQN model

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    
    G_0, step = [], 0
    for e in range(num_episodes):  
        s, done, trunc, g = env.reset()[0], False, False, 0.

        while not (done | trunc):
            #build training batch
            a = qr.max_action(torch.tensor([s]), epsilon=eps(step)).item()
            s_prime, r, done, trunc, _ = env.step(a)
            memory.push(s, a, s_prime, r, float(done))
            
            g = g * gamma + r
            s = s_prime

            #process batch and update network
            if len(memory) < batch_size: continue
            batch = memory.sample(batch_size)
            states, actions, rewards, next_states, dones = batch

            z_theta = qr_target(next_states)[np.arange(batch_size), qr_target.max_action(next_states)]
            qr.update(batch, z_theta.detach(), gamma)
            
            #refresh target network and log results
            if step % refresh_rate == 0: qr_target.load_state_dict(qr.state_dict())
            step += 1
        G_0.append(g)
        
        if e % eval_every == 0: 
            eval = evaluate(env, qr, gamma, num_episodes = 3)
            print("Eval Return Episode {}:".format(e), round(eval.mean(), 2), end = " | ")
            print("Actual Return Episode {}".format(e), round(np.mean(G_0), 2), end='\n')
        
    return qr, G_0

def train_nstep_qr(
    env, #open-ai environment
    qr:QrNetwork,
    qr_target:QrNetwork,
    writer:SummaryWriter,
    gamma:float,
    num_episodes:int,
    batch_size:int,
    eval_every:int,
    n_step:int,
    eps:Callable,
    log = True,
    soft = False,
    tau = .5,
    memory = ReplayMemory(10000),
    refresh_rate = 100,
    importance_correction = .5,
    ) -> Iterable[float]:
    """
    implement simple verison of QR-DQN model

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    
    G_0, step = [], 0
    reward_decay = np.array([pow(gamma, n) for n in range(n_step)])

    for e in range(num_episodes):  
        s, done, trunc, t = env.reset()[0], False, False, step
        g, past_states, past_actions, sample_index = [], [], [], []
        
        while not (done | trunc):
            #build training batch
            a = qr.max_action(torch.tensor([s]), epsilon=eps(step)).item()
            s_prime, r, done, trunc, _ = env.step(a)
            
            past_states.append(s)
            past_actions.append(a)
            g.append(r)
            s = s_prime

            tau = step - t + 1
            if tau >= n_step:
                ret = (reward_decay * g[-n_step:]).sum()
                s_tau, a_tau = past_states[-n_step], past_actions[-n_step]
                memory.push(s_tau, a_tau, s_prime, ret, float(done))
            step += 1

            #process batch and update network
            if len(memory) < batch_size: continue
            if memory.type == 'priority': batch, weights, sample_index = memory.sample(batch_size, beta = importance_correction)
            else: batch = memory.sample(batch_size)
            
            states, actions, returns, next_states, dones = batch
            z_theta = qr_target(next_states)[np.arange(batch_size), qr_target.max_action(next_states)]
            td, loss = qr.update(batch, z_theta.detach(), gamma, n_step)
            writer.add_scalar('Loss', loss, step)
            if memory.type == 'priority': memory.update_priorities(sample_index, td)
            
            #refresh target network and log results
            if step % refresh_rate == 0: 
                if soft: soft_update(qr, qr_target, tau)
                else: qr_target.load_state_dict(qr.state_dict())

            
        G_0.append(sum(g))
        
        if e % eval_every == 0: 
            eval_batch = evaluate(env, qr, gamma, num_episodes = 3)
            writer.add_scalar("Reward", np.mean(eval_batch), e)

            if log:
                print("Eval Return Episode {}:".format(e), round(np.mean(eval_batch), 2), end = " | ")
                print("Actual Return Episode {}".format(e), round(np.mean(G_0), 2), end='\n')
        
    return qr, G_0

def train_concurrent_qr(
    env, #open-ai environment
    qr:QrNetwork,
    qr_target:QrNetwork,
    risk_functions:RiskFunction,
    gamma:float,
    num_actors:int,
    iterations:int,
    batch_size:int,
    eval_every:int,
    eps:Callable,
    memory = ReplayMemory(10000),
    refresh_rate = 100,
    ) -> Iterable[float]:
    """
    QR-DQN model with multiple agents with different risk preferences

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """

    multi_env = MultiStep(env, qr, num_actors, risk_functions, memory, eps)
    step = 0
    for i in range(iterations):  
        multi_env.gen()

        #process batch and update network
        if len(memory) < batch_size: continue
        for n in range(1):
            batch = memory.sample(batch_size)
            states, actions, rewards, next_states, dones = batch

            z_theta = qr_target(next_states)[np.arange(batch_size), qr_target.max_action(next_states)]
            qr.update(batch, z_theta.detach(), gamma)
                
            #refresh target network and log results
            if step % refresh_rate == 0: qr_target.load_state_dict(qr.state_dict())
        step += 1
        
        if i % eval_every == 0: 
            eval = evaluate(env, qr, gamma, num_episodes = 3)
            print("Eval Return Iteration {}:".format(i), round(eval.mean(), 2))
        
    return qr

def evaluate(
    env, #open-ai environment
    qr:QrNetwork,
    gamma:float,
    num_episodes = 10,
    w = np.zeros(0),
) -> float:
    
    G_0 = []
    for _ in range(num_episodes):  
        s, done, trunc, g = env.reset()[0], False, False, 0.

        while not (done | trunc):
            if len(w): a = qr.max_biased_action(torch.tensor([s]), w=w, epsilon=0).item()
            else: a = qr.max_action(torch.tensor([s]), epsilon=0).item()
            s_prime, r, done, trunc, _ = env.step(a)
            
            g = g * gamma + r
            s = s_prime

        G_0.append(g)
    
    return pd.Series(G_0)


if __name__ == "__main__":
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)

    N = 32
    alpha = 1e-3
    num_episodes = 2500
    #replay = PriorityReplayMemory(capacity=10000, alpha=.7)
    replay = ReplayMemory(capacity=10000)
    writer = SummaryWriter("logs/" + "risk_bias_test_two")

    #test biased action selection
    qr = BiasedQr(state_dims=env.observation_space.shape[0], num_actions=env.action_space.n, n=N, alpha=alpha, decay = -.15, clip = 0)
    qr_target = BiasedQr(state_dims=env.observation_space.shape[0], num_actions=env.action_space.n, n=N, alpha=alpha, decay = .15, clip = 0)

    eps_start, eps_end, eps_dec = 0.9, 0.05, 500
    eps_decay = lambda steps: eps_end + (eps_start - eps_end) * np.exp(-1. * steps / eps_dec)
    eps = lambda steps: .05

    qr, G = train_nstep_qr(env, qr, qr_target, writer, gamma=1., num_episodes=num_episodes, batch_size=128, eval_every=50, 
                           eps=eps, n_step=1, memory=replay, soft=False)
    G_test = evaluate(env, qr, gamma=1, num_episodes=100)
    print(G_test.describe())

    #test regular action selection
    replay = ReplayMemory(capacity=10000)
    writer = SummaryWriter("logs/" + "test_one")
    qr = QrNetwork(state_dims=env.observation_space.shape[0], num_actions=env.action_space.n, n=N, alpha=alpha)
    qr_target = QrNetwork(state_dims=env.observation_space.shape[0], num_actions=env.action_space.n, n=N, alpha=alpha)

    qr, G = train_nstep_qr(env, qr, qr_target, writer, gamma=1., num_episodes=num_episodes, batch_size=64, eval_every=50, 
                           eps=eps_decay, n_step=1, memory=replay, soft=False)

    G_test = evaluate(env, qr, gamma=1, num_episodes=100)
    print(G_test.describe())

    exit()

    #test n-step return
    n_step = 4
    eps = lambda steps: .01
    writer = SummaryWriter("logs/" + "{}_step_test_one".format(n_step))

    qr = QrNetwork(state_dims=env.observation_space.shape[0], num_actions=env.action_space.n, n=N, alpha=alpha)
    qr_target = QrNetwork(state_dims=env.observation_space.shape[0], num_actions=env.action_space.n, n=N, alpha=alpha)

    qr, G = train_nstep_qr(env, qr, qr_target, writer, gamma=1., num_episodes=num_episodes, batch_size=64, eval_every=50, 
                           eps=eps, n_step=n_step)

    G_test = evaluate(env, qr, gamma=1, num_episodes=100)
    print("Eval for {}-step return\n".format(n_step), G_test.describe())

    


    #test concurrent implementation
    num_actors = 1
    equal_weight = lambda steps: 0
    risk_function = RiskFunction(init=0, function=equal_weight, n=N, period=1, update_freq=10000)
    risk_funcs = [risk_function] * num_actors

    qr = QrNetwork(state_dims=env.observation_space.shape[0], num_actions=env.action_space.n, n=N, alpha=alpha)
    qr_target = QrNetwork(state_dims=env.observation_space.shape[0], num_actions=env.action_space.n, n=N, alpha=alpha)

    qr = train_concurrent_qr(env, qr, qr_target, risk_funcs, gamma=1., iterations=200000, num_actors=num_actors,
                             batch_size=64, eval_every=20000, eps=eps)
    G_test = evaluate(env, qr, gamma=1, num_episodes=100)
    print(G_test.describe())






    
