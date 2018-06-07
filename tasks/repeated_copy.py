import random
import torch
import numpy as np
import torch.optim as optim
from torch import nn
from ntm.factory import ntm_factory
from trainer import train
from utils import plot


class Model:
    def __init__(self, ntm, data_loader, optimizer, criterion, task_name):
        self.ntm = ntm
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.task_name = task_name


class DataLoader:
    def __init__(self,
                 n_batches,
                 batch_size,
                 input_size,
                 min_len,
                 max_len,
                 min_repeat,
                 max_repeat):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.input_size = input_size
        self.min_len = min_len
        self.max_len = max_len
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat

    def normalize_repeat(self, repeat):
        repeat_mean = (self.max_repeat + self.min_repeat) / 2
        repeat_var = (((self.max_repeat - self.min_repeat + 1) ** 2) - 1) / 12
        repeat_std = np.sqrt(repeat_var)
        return (repeat - repeat_mean) / repeat_std

    def gen_data(self):
        for batch_num in range(self.n_batches):
            # All batches have the same sequence length
            seq_len = random.randint(self.min_len, self.max_len)
            n_repeats = random.randint(self.min_repeat, self.max_repeat)

            seq = np.random.binomial(1, 0.5, (seq_len, self.batch_size, self.input_size))
            seq = torch.from_numpy(seq)

            # The input includes an additional channel used for the delimiter
            inp = torch.zeros(seq_len + 2, batch_size, self.input_size + 2)
            inp[:seq_len, :, :self.input_size] = seq
            inp[seq_len, :, self.input_size] = 1.0  # delimiter in our control channel
            inp[seq_len + 1, :, self.input_size + 1] = self.normalize_repeat(n_repeats)

            out = torch.zeros(seq_len * n_repeats + 1, batch_size, input_size + 1)
            out[:seq_len * n_repeats, :, :input_size] = seq.clone().repeat(n_repeats, 1, 1)
            out[seq_len * n_repeats, :, input_size] = 1.0

            yield batch_num, inp.float(), out.float()


def final_eval(ntm, input_size, seq_len, n_repeats, data_loader):
    ntm.init_state(1)

    seq = np.random.binomial(1, 0.5, (seq_len, 1, input_size))
    seq = torch.from_numpy(seq)

    # The input includes an additional channel used for the delimiter
    inp = torch.zeros(seq_len + 2, 1, input_size + 2)
    inp[:seq_len, :, :input_size] = seq
    inp[seq_len, :, input_size] = 1.0  # delimiter in our control channel
    inp[seq_len + 1, :, input_size + 1] = data_loader.normalize_repeat(n_repeats)
    inp_seq_len = inp.size(0)

    correct_out = torch.zeros(seq_len * n_repeats + 1, 1, input_size + 1)
    correct_out[:seq_len * n_repeats, :, :input_size] = seq.clone().repeat(n_repeats, 1, 1)
    correct_out[seq_len * n_repeats, :, input_size] = 1.0

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
    ntm = ntm_factory(input_size + 2, input_size + 1,
                      100, 1,
                      1, 1,
                      128, 20)

    data_loader = DataLoader(50000, batch_size, input_size, 1, 10, 1, 10)

    model = Model(ntm,
                  data_loader,
                  optim.RMSprop(ntm.parameters(), lr=0.0001, alpha=0.95, momentum=0.9),
                  nn.BCELoss(),
                  'repeated_copy')
    # ntm = train(model)
    inp, correct_out, y_out = final_eval(ntm, 8, 5, 3, data_loader)
    plot(inp, correct_out, y_out)
