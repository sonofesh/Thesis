from torch.utils.tensorboard import SummaryWriter
from sympy import *

import torch.nn as nn
import numpy as np
import torch

class QrNetwork(nn.Module):
    def __init__(self, state_dims, num_actions, n, alpha, clip = 0, layer_width = 64):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        super(QrNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dims, layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_width, layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_width, num_actions * n)
        )
        #self.model.apply(init_weights)

        self.num_actions = num_actions
        self.n = n
        self.clip = clip

        self.quantiles = torch.Tensor(np.arange(1/(2 * n), 1, 1/n)) #ex [.25, .75] for n = 2
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr = alpha, betas=(.9,.999))

    def forward(self, states):
        self.model.eval()
        return self.model(states).view(-1, self.num_actions, self.n)

    def max_action(self, states, epsilon=0):
        """
        input: 
            states: batch of states or [state]
            epsilon: epsilon
        output: action
        """
        action_values = self.forward(states).sum(axis = 2)
        
        if np.random.random() >= epsilon: return torch.max(action_values, 1)[1]
        else: return torch.randint(high=self.num_actions, size=(len(states),))
    
    def max_biased_action(self, states, w, epsilon=0):
        action_values = (self.forward(states) * w).sum(axis = 2)
        
        if np.random.random() >= epsilon: return torch.max(action_values, 1)[1]
        else: return torch.randint(high=self.num_actions, size=(len(states),))

    def update(self, batch, z_theta, gamma, imp_weights=[], n_step=1):
        """
        batch: s, a, r, s_, done
        gamma: gamma
        """
        states, actions, returns, next_states, dones = batch
        theta = self.forward(states)[np.arange(len(states)), actions]
        t_theta = returns + pow(gamma, n_step) * (1 - dones) * z_theta

        td = t_theta.t().unsqueeze(-1) - theta 
        loss = huber(td) * (self.quantiles - (td.detach() < 0).float()).abs()
        if self.clip: loss = torch.clamp(loss.mean(), min=-0, max=1)
        else: loss = loss.mean()

        self.model.train()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #move batch_num to first dim and take mean of quantiles to get td error
        return torch.movedim(td, 1, 0).mean(dim=(1, 2)).abs(), loss


class BiasedQr(QrNetwork):
    def __init__(self, state_dims, num_actions, n, alpha, decay, clip):
        super().__init__(state_dims, num_actions, n, alpha, clip)
        self._set_decay(decay=decay)
    
    def max_action(self, states, epsilon=0):
        action_values = (self.forward(states) * self.w).sum(axis = 2)
        
        if np.random.random() >= epsilon: return torch.max(action_values, 1)[1]
        else: return torch.randint(high=self.num_actions, size=(len(states),))
    
    def load_state_dict(self, state_dict):
        #update main network wieghts but keep target the same
        w = self.w
        super().load_state_dict(state_dict)
        self.w = w
    
    def _set_decay(self, decay):
        self.decay = decay
        self.w = torch.Tensor(get_exp_weights(decay, self.n))
    
class RandomBiasQr(BiasedQr):
    def __init__(self, state_dims, num_actions, n, alpha, decay, clip):
        super().__init__(state_dims, num_actions, n, alpha)
        self._set_decay(decay=decay)
    
    def max_action(self, states, epsilon=0):
        action_values = (self.forward(states) * self.w).sum(axis = 2)
        
        if np.random.random() >= epsilon: return torch.max(action_values, 1)[1]
        else: return torch.randint(high=self.num_actions, size=(len(states),))
    
    def load_state_dict(self, state_dict):
        #update main network wieghts but keep target the same
        w = self.w
        super().load_state_dict(state_dict)
        self.w = w
    
    def _set_decay(self, decay):
        self.decay = decay
        self.w = torch.Tensor(get_exp_weights(decay, self.n))

def get_exp_weights(rate, n):
    t = Symbol('t')
    if rate == 0: return np.full(n, 1./n)
    y = rate * pow(np.e, -rate * t)

    f = lambdify(t, y, 'numpy')
    w = np.ones(n).cumsum()
    return f(w)/f(w).sum()


def soft_update(model, target_model, tau):
    """Soft update model parameters. θ_target = τ*θ_local + (1 - τ)*θ_target
        tau (float): interpolation parameter 
    """
    for target_param, model_param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau * model_param.data + (1.0 - tau) * target_param.data)

def huber(x, k = 1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))