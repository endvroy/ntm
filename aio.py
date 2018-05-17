import torch
from torch import nn

from .ntm import NTM
from .controller import Controller
from .head import ReadHead, WriteHead
from .memory import Memory


def ntm_factory(input_size, output_size,
                controller_out_size, controller_layers,
                n_read_heads, n_write_heads,
                N, M):
    input_size = input_size
    output_size = output_size

    memory = Memory(N, M)
    read_heads = nn.ModuleList()
    for i in range(n_read_heads):
        read_heads.append(ReadHead(M, controller_out_size))

    write_heads = nn.ModuleList()
    for i in range(n_write_heads):
        write_heads.append(WriteHead(M, controller_out_size))

    controller = Controller(input_size, controller_out_size, controller_layers)

    ntm = NTM(input_size, output_size, controller, memory, read_heads, write_heads)
    return ntm
