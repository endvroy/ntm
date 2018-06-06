import matplotlib.pyplot as plt
import torch


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


def calculate_num_params(net):
    """Returns the total number of parameters."""
    num_params = 0
    for p in net.parameters():
        num_params += p.data.view(-1).size(0)
    return num_params


def binarize(tensor):
    binarized = tensor.clone().data
    binarized.apply_(lambda x: 0 if x < 0.5 else 1)
    return binarized
