import torch
from torch import nn
from torch.nn import Parameter
import numpy as np


class Controller(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super(Controller, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=output_size,
                            num_layers=num_layers)

        self.reset()

    def reset(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs + self.num_outputs))
                nn.init.uniform(p, -stdev, stdev)

    def forward(self, x, prev_state):
        out, state = self.lstm(x.unsqueeze(0), prev_state)
        return out.squeeze(0), state
