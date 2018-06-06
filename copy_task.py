import random
import torch
from aio import ntm_factory
import numpy as np


def gen_data(n_batches,
             batch_size,
             input_size,
             min_len,
             max_len):
    """Generator of random sequences for the copy task.

    Creates random batches of "bits" sequences.

    All the sequences within each batch have the same length.
    The length is [`min_len`, `max_len`]

    :param n_batches: Total number of batches to generate.
    :param input_size: The width of each item in the sequence.
    :param batch_size: Batch size.
    :param min_len: Sequence minimum length.
    :param max_len: Sequence maximum length.

    NOTE: The input width is `input_size + 1`, the additional input
    contain the delimiter.
    """
    for batch_num in range(n_batches):
        # All batches have the same sequence length
        seq_len = random.randint(min_len, max_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, input_size))
        seq = torch.from_numpy(seq)

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len + 1, batch_size, input_size + 1)
        inp[:seq_len, :, :input_size] = seq
        inp[seq_len, :, input_size] = 1.0  # delimiter in our control channel
        out = seq.clone()

        yield batch_num, inp.float(), out.float()


import torch.optim as optim
from torch import nn
import math


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


def train():
    input_size = 8
    batch_size = 1
    ntm = ntm_factory(input_size + 1, input_size,
                      100, 1,
                      1, 1,
                      128, 20)

    optimizer = optim.RMSprop(ntm.parameters(), lr=0.0001, alpha=0.95, momentum=0.9)
    criterion = nn.BCELoss()

    for i, inp, correct_out in gen_data(50000, batch_size, input_size, 1, 20):
        loss = train_batch(ntm, batch_size, inp, correct_out, optimizer, criterion)
        if i % 1000 == 0:
            print(f'{i} batches finished, loss={loss}')
            torch.save(ntm.state_dict(), f'checkpoints/{i}.sd')
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


def debug():
    input_size = 8
    batch_size = 1
    ntm = ntm_factory(input_size + 1, input_size,
                      100, 1,
                      1, 1,
                      128, 20)
    ntm.init_state(batch_size)
    # print(calculate_num_params(ntm))
    return ntm


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
    ntm = train()
    # inp, correct_out, y_out = final_eval(ntm, 8, 20)
    plot(inp, correct_out, y_out)
