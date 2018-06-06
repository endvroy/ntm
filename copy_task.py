import random
import torch
import numpy as np
import torch.optim as optim
from torch import nn
from factory import ntm_factory
from trainer import train
from utils import plot


class Model:
    def __init__(self, ntm, data_loader, optimizer, criterion):
        self.ntm = ntm
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion


class DataLoader:
    def __init__(self,
                 n_batches,
                 batch_size,
                 input_size,
                 min_len,
                 max_len):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.input_size = input_size
        self.min_len = min_len
        self.max_len = max_len

    def gen_data(self):
        for batch_num in range(self.n_batches):
            # All batches have the same sequence length
            seq_len = random.randint(self.min_len, self.max_len)
            seq = np.random.binomial(1, 0.5, (seq_len, self.batch_size, self.input_size))
            seq = torch.from_numpy(seq)

            # The input includes an additional channel used for the delimiter
            inp = torch.zeros(seq_len + 1, batch_size, self.input_size + 1)
            inp[:seq_len, :, :self.input_size] = seq
            inp[seq_len, :, self.input_size] = 1.0  # delimiter in our control channel
            out = seq.clone()

            yield batch_num, inp.float(), out.float()


def final_eval(ntm, input_size, seq_len):
    ntm.init_state(1)
    seq = np.random.binomial(1, 0.5, (seq_len, 1, input_size))
    seq = torch.from_numpy(seq)

    # The input includes an additional channel used for the delimiter
    inp = torch.zeros(seq_len + 1, 1, input_size + 1)
    inp[:seq_len, :, :input_size] = seq
    inp[seq_len, :, input_size] = 1.0  # delimiter in our control channel
    inp_seq_len = inp.size(0)
    correct_out = seq.clone()

    out_seq_len, batch_size, _ = correct_out.size()
    y_out = torch.zeros(correct_out.size())
    with torch.no_grad():
        for j in range(inp_seq_len):
            out = ntm(inp[j])

        for j in range(out_seq_len):
            y_out[j] = ntm()

    return inp, correct_out, y_out


if __name__ == '__main__':
    input_size = 8
    batch_size = 1
    ntm = ntm_factory(input_size + 1, input_size,
                      100, 1,
                      1, 1,
                      128, 20)

    data_loader = DataLoader(10000, batch_size, input_size, 1, 20)

    model = Model(ntm,
                  data_loader,
                  optim.RMSprop(ntm.parameters(), lr=0.0001, alpha=0.95, momentum=0.9),
                  nn.BCELoss())
    ntm = train(model)
    inp, correct_out, y_out = final_eval(ntm, 8, 80)
    plot(inp, correct_out, y_out)
