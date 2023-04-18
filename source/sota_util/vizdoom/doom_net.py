"""
Vincent Lombardi
"""

import torch
import torch.nn as nn

import torch.nn.functional as F

from .myutil import set_init



class Net(nn.Module):

    def __init__(self, s_dim, a_dim, hidden_size, h_size):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.actor1 = nn.Linear(s_dim, hidden_size)
        self.actor2 = nn.Linear(hidden_size, h_size)
        self.actor3 = nn.Linear(h_size, h_size)
        self.actor4 = nn.Linear(h_size, a_dim)

        self.critic1 = nn.Linear(s_dim, hidden_size)
        self.critic2 = nn.Linear(hidden_size, h_size)
        self.critic3 = nn.Linear(h_size, h_size)
        self.critic4 = nn.Linear(h_size, 1)
        set_init([self.actor1, self.actor2, self.actor3, self.actor4, self.critic1, self.critic2, self.critic3, self.critic4])

        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        actor1 = F.leaky_relu(self.actor1(x))
        critic1 = F.leaky_relu(self.critic1(x))
        actor2 = F.leaky_relu(self.actor2(actor1))
        critic2 = F.leaky_relu(self.critic2(critic1))
        actor3 = F.leaky_relu(self.actor3(actor2))
        critic3 = F.leaky_relu(self.critic3(critic2))
        logits = self.actor4(actor3)
        values = self.critic4(critic3)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]


