import torch
from torch import nn
import torch.nn.functional as F


class NTM(nn.Module):
    def __init__(self, input_size, output_size, controller, memory, read_heads, write_heads):
        super(NTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.controller = controller
        self.memory = memory
        self.read_heads = read_heads
        self.write_heads = write_heads

        _, M = memory.size()

        self.output_fc = nn.Linear(controller.output_size + M * len(self.read_heads), output_size)
        self.reset_params()

    def reset_params(self):
        # Initialize the linear layer
        nn.init.xavier_uniform_(self.output_fc.weight, gain=1)
        nn.init.normal_(self.output_fc.bias, std=0.01)

    def init_state(self, batch_size):
        self.memory.reset(batch_size)
        N, M = self.memory.size()
        init_prev_reads = [(torch.randn(1, M) * 0.01).repeat(batch_size, 1)
                           for i in range(len(self.read_heads))]
        init_controller_state = self.controller.new_init_state(batch_size)
        prev_read_weights = torch.zeros(batch_size, N)
        prev_write_weights = torch.zeros(batch_size, N)
        return init_prev_reads, init_controller_state, prev_read_weights, prev_write_weights

    def forward(self,
                inp,
                prev_reads,
                prev_controller_state,
                prev_read_weights,
                prev_write_weights):
        controller_outs, controller_state = self.controller(torch.cat([inp] + prev_reads, dim=1),
                                                            prev_controller_state)

        read_weights = []
        reads = []
        for prev_weight, head in zip(prev_read_weights, self.read_heads):
            params = head(controller_outs)
            weight = self.memory.address(prev_weight, *params)
            read_weights.append(weight)
            read_vec = self.memory.read(weight)
            reads.append(read_vec)

        write_weights = []
        for prev_weight, head in zip(prev_write_weights, self.write_heads):
            params = head(controller_outs)
            *addr_params, e, a = params
            weight = self.memory.address(prev_weight, *addr_params)
            write_weights.append(weight)
            self.memory.write(weight, e, a)

        out = F.sigmoid(self.output_fc(torch.cat([controller_outs] + reads, dim=1)))

        return out, reads, controller_state, read_weights, write_weights
