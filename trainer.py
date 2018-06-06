import torch
import pathlib
import datetime


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


def save_checkpoint(ntm, task_name, timestamp, i):
    pathlib.Path(f'checkpoints/{task_name}/{timestamp}/').mkdir(parents=True, exist_ok=True)
    torch.save(ntm.state_dict(), f'checkpoints/{task_name}/{timestamp}/{i}.sd')


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
            save_checkpoint(ntm, 'copy_task', now, i + 1)
    return ntm
