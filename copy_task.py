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
        train_batch(ntm, batch_size, inp, correct_out, optimizer, criterion)

    return ntm


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
    print(calculate_num_params(ntm))


if __name__ == '__main__':
    train()
