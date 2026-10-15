"""Small CPU/Gloo gradient reductions shared by the two trainers."""
import torch

def all_sum(values, world):
    if world > 1:
        torch.distributed.all_reduce(values)
    return values


def reduce_gradients(params, count, world):
    """FP32 sum over active local groups (not windows); None stays None everywhere."""
    if world == 1:
        for parameter in params:
            if parameter.grad is not None:
                parameter.grad.div_(count)
        return
    present = all_sum(torch.tensor([float(p.grad is not None) for p in params], dtype=torch.float32), world)
    flat = all_sum(torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().cpu()
                              .flatten() for p in params]), world).div_(count)
    offset = 0
    for parameter, seen in zip(params, present.tolist()):
        n = parameter.numel()
        parameter.grad = flat[offset:offset + n].view_as(parameter).to(parameter.device, parameter.dtype) \
            if seen else None
        offset += n


def gather(value, world):
    if world == 1:
        return [value]
    parts = [None] * world
    torch.distributed.all_gather_object(parts, value)
    return parts


