# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle


def split(x, *args, **kwargs):
    if args:
        if len(args) == 1:
            return paddle.split(x, x.shape[0] // args[0])
        else:
            return paddle.split(x, x.shape[args[1]] // args[0], args[1])
    elif kwargs:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")
            kwargs["num_or_sections"] = x.shape[kwargs["axis"]] // kwargs.pop(
                "split_size"
            )
        else:
            kwargs["num_or_sections"] = x.shape[0] // kwargs.pop("split_size")
        return paddle.split(x, **kwargs)


def paddle_unfold(tensor, dimension, size, step=1):
    assert dimension < len(
        tensor.shape
    ), "dimension must be less than tensor dimensions"
    assert (
        tensor.shape[dimension] >= size
    ), "size should not be greater than the dimension of tensor"

    slices = []
    for i in range(0, tensor.shape[dimension] - size + 1, step):
        start = [0] * len(tensor.shape)
        end = list(tensor.shape)
        start[dimension] = i
        end[dimension] = i + size
        axes = list(range(len(start)))
        slice = paddle.slice(tensor, axes, start, end)
        slices.append(slice)

    unfolded_tensor = paddle.stack(slices, axis=dimension)

    # The dimension is converted to the last dimension
    perm = list(range(len(tensor.shape)))
    for i, j in enumerate(perm):
        if i > dimension:
            perm[i] = j + 1
    perm.append(dimension + 1)
    unfolded_tensor = unfolded_tensor.transpose(perm)

    return unfolded_tensor


# torch and paddle may have diff
def pad(x, pad, mode="constant", value=0.0):
    assert len(x.shape) * 2 != len(pad)
    return paddle.nn.functinal.pad(x, pad, mode, value)

def premute(x, *args, **kwargs):
    return paddle.transpose(x, *args, *kwargs)
