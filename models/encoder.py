import math
import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(hidden_size, 64)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(64, 32)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(32, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.Sigmoid()

    def forward(self, x, mask_cls):
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.hidden3(x).squeeze(-1)
        x = self.act3(x) * mask_cls.float()
        return x
