import sys
sys.path.append('/nfs/github/recurrent/out/utils')
import paddle_aux
import paddle
"""
Various utilities for neural networks.
"""
import math


class SiLU(paddle.nn.Layer):

    def forward(self, x):
        return x * paddle.nn.functional.sigmoid(x=x)


class GroupNorm32(paddle.nn.GroupNorm):

    def forward(self, x):
        return super().forward(x.astype(dtype='float32')).astype(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
>>>        return torch.nn.Conv1d(*args, **kwargs)
    elif dims == 2:
>>>        return torch.nn.Conv2d(*args, **kwargs)
    elif dims == 3:
>>>        return torch.nn.Conv3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
>>>    return torch.nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
>>>        return torch.nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
>>>        return torch.nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
>>>        return torch.nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().scale_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(axis=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = paddle.exp(x=-math.log(max_period) * paddle.arange(start=0, end
        =half).astype('float32') / half).to(device=timesteps.place)
    args = timesteps[:, (None)].astype(dtype='float32') * freqs[None]
    embedding = paddle.concat(x=[paddle.cos(x=args), paddle.sin(x=args)],
        axis=-1)
    if dim % 2:
        embedding = paddle.concat(x=[embedding, paddle.zeros_like(x=
            embedding[:, :1])], axis=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with paddle.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        out_5 = x.detach()
        out_5.stop_gradient = not True
        ctx.input_tensors = [out_5 for x in ctx.input_tensors]
        with paddle.enable_grad():
            shallow_copies = [x.reshape(x.shape) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = paddle.grad(outputs=output_tensors, inputs=ctx.
            input_tensors + ctx.input_params, grad_outputs=output_grads,
            allow_unused=True)
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
