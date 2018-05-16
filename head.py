import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


class ReadHead(nn.Module):
    def __init__(self, M, controller_out_size):
        super(ReadHead, self).__init__()
        self.M = M
        self.param_sizes = [self.M, 1, 1, 3, 1]  # k, beta, g, s, y
        self.fc = nn.Linear(controller_out_size, sum(self.param_sizes))
        self.reset()
        self.is_read_head = True

    def reset(self):
        # Initialize the linear layers
        nn.init.xavier_uniform(self.fc.weight, gain=1.4)
        nn.init.normal(self.fc.bias, std=0.01)

    def forward(self, controller_output):
        params = self.fc(controller_output)
        k, beta, g, s, y = _split_cols(params, self.param_sizes)
        beta = F.softplus(beta)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        y = 1 + F.softplus(y)

        return k, beta, g, s, y


class WriteHead(nn.Module):
    def __init__(self, M, controller_out_size):
        super(WriteHead, self).__init__()
        self.M = M
        self.param_sizes = [self.M, 1, 1, 3, 1, self.M, self.M]  # k, beta, g, s, y
        self.fc = nn.Linear(controller_out_size, sum(self.param_sizes))
        self.reset()
        self.is_read_head = False

    def reset(self):
        # Initialize the linear layers
        nn.init.xavier_uniform(self.fc.weight, gain=1.4)
        nn.init.normal(self.fc.bias, std=0.01)

    def forward(self, controller_output):
        params = self.fc(controller_output)
        k, beta, g, s, y, e, a = _split_cols(params, self.param_sizes)
        beta = F.softplus(beta)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        y = 1 + F.softplus(y)
        e = F.sigmoid(e)

        return k, beta, g, s, y, e, a
