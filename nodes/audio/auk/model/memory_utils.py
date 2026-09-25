import torch
import torch.nn.functional as F


def fuse_layers(states, logits, scale):
    """Fuse encoder layers without allocating a second stacked hidden-state tensor."""
    device = states[0].device
    weights = logits.to(device).softmax(dim=0)
    hidden = torch.zeros_like(states[1], dtype=torch.float32)
    for weight, state in zip(weights, states[1:]):
        hidden = hidden + F.layer_norm(state, [state.shape[-1]]).float() * weight
    return hidden * scale.to(device)


def euler_final(fn, initial, times, progress_cb=None):
    """Same fixed-grid Euler update as torchdiffeq; discard intermediate states.

    progress_cb（sfnodes 扩展，可选）：每步后回调 (done, total)，供节点进度条/中断检查。
    """
    value = initial
    total = len(times) - 1
    for done, (start, end) in enumerate(zip(times[:-1], times[1:]), start=1):
        value = value + (end - start) * fn(start, value)
        if progress_cb is not None:
            progress_cb(done, total)
    return value
