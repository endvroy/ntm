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

        self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.output_size) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.output_size) * 0.05)

    def new_init_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def reset(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.input_size + self.output_size))
                nn.init.uniform_(p, -stdev, stdev)

    def forward(self, x, prev_state):
        out, state = self.lstm(x.unsqueeze(0), prev_state)
        return out.squeeze(0), state
