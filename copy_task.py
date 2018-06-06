import random
import torch
from factory import ntm_factory
import numpy as np


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


import torch.optim as optim
from torch import nn


def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    params = list(filter(lambda p: p.grad is not None, net.parameters()))
    # params = [x for x in net.parameters() if x.grad is not None]
    for p in params:
        p.grad.data.clamp_(-10, 10)


def train_batch(ntm, batch_size, inp, correct_out, optimizer, criterion):
    optimizer.zero_grad()
    ntm.init_state(batch_size)

    inp_seq_len = inp.size(0)
    out_seq_len, batch_size, _ = correct_out.size()
    for j in range(inp_seq_len):
        out = ntm(inp[j])

    y_out = torch.zeros(correct_out.size())
    for j in range(out_seq_len):
        y_out[j] = ntm()

    loss = criterion(y_out, correct_out)
    loss.backward()
    clip_grads(ntm)
    optimizer.step()
    return loss.data.item()


import pathlib


def save_checkpoint(ntm, task_name, timestamp, i):
    pathlib.Path(f'checkpoints/{task_name}/{timestamp}/').mkdir(parents=True, exist_ok=True)
    torch.save(ntm.state_dict(), f'checkpoints/{task_name}/{timestamp}/{i}.sd')


import datetime


def train(model):
    ntm = model.ntm
    optimizer = model.optimizer
    criterion = model.criterion
    data_loader = model.data_loader

    now = datetime.datetime.now().isoformat()

    for i, inp, correct_out in data_loader.gen_data():
        loss = train_batch(ntm, data_loader.batch_size, inp, correct_out, optimizer, criterion)
        if (i + 1) % 1000 == 0:
            print(f'{i + 1} batches finished, loss={loss}')
            save_checkpoint(ntm, 'copy_task', now, i)
    return ntm


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


def calculate_num_params(net):
    """Returns the total number of parameters."""
    num_params = 0
    for p in net.parameters():
        num_params += p.data.view(-1).size(0)
    return num_params


import matplotlib.pyplot as plt


def plot(inp, correct_out, y_out):
    plt.figure(1)
    ax1 = plt.subplot(411)
    plt.title('input')
    plt.imshow(torch.t(inp.squeeze(1)), vmin=0, vmax=1, cmap='Blues')
    ax2 = plt.subplot(412)
    plt.title('correct output')
    plt.imshow(torch.t(correct_out.squeeze(1)), vmin=0, vmax=1, cmap='Blues')
    ax3 = plt.subplot(413)
    plt.title('actual output')
    plt.imshow(torch.t(y_out.squeeze(1)), vmin=0, vmax=1, cmap='Blues')
    error = torch.abs(correct_out.float() - y_out)
    ax4 = plt.subplot(414)
    plt.title('error')
    plt.imshow(torch.t(error.squeeze(1)), vmin=0, vmax=1, cmap='Reds')
    # plt.colorbar()
    plt.show()


def binarize(tensor):
    binarized = tensor.clone().data
    binarized.apply_(lambda x: 0 if x < 0.5 else 1)
    return binarized


if __name__ == '__main__':
    input_size = 8
    batch_size = 1
    ntm = ntm_factory(input_size + 1, input_size,
                      100, 1,
                      1, 1,
                      128, 20)

    data_loader = DataLoader(20000, batch_size, input_size, 1, 20)

    model = Model(ntm,
                  data_loader,
                  optim.RMSprop(ntm.parameters(), lr=0.0001, alpha=0.95, momentum=0.9),
                  nn.BCELoss())
    ntm = train(model)
    inp, correct_out, y_out = final_eval(ntm, 8, 80)
    plot(inp, correct_out, y_out)
