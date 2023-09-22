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


class Generator(paddle.nn.Layer):
    def __init__(
        self, latent_dim=100, out_channels=1, out_scale=0.01, sample_mode="bilinear"
    ):
        super(Generator, self).__init__()
        self.out_scale = out_scale
        self.init_size = 32 // 4
        self.l1 = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=latent_dim, out_features=128 * self.init_size**2
            )
        )
        self.conv_blocks = paddle.nn.Sequential(
            paddle.nn.BatchNorm2D(num_features=128),
            paddle.nn.Upsample(scale_factor=2, mode=sample_mode),
            paddle.nn.Conv2D(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.BatchNorm2D(num_features=128, epsilon=0.8),
            paddle.nn.LeakyReLU(negative_slope=0.2),
            paddle.nn.Upsample(scale_factor=2, mode=sample_mode),
            paddle.nn.Conv2D(
                in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.BatchNorm2D(num_features=64, epsilon=0.8),
            paddle.nn.LeakyReLU(negative_slope=0.2),
            paddle.nn.Upsample(scale_factor=2, mode=sample_mode),
            paddle.nn.Conv2D(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.BatchNorm2D(num_features=64, epsilon=0.8),
            paddle.nn.LeakyReLU(negative_slope=0.2),
            paddle.nn.Upsample(scale_factor=2, mode=sample_mode),
            paddle.nn.Conv2D(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.BatchNorm2D(num_features=32, epsilon=0.8),
            paddle.nn.LeakyReLU(negative_slope=0.2),
            paddle.nn.Upsample(scale_factor=2, mode=sample_mode),
            paddle.nn.Conv2D(
                in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.BatchNorm2D(num_features=16, epsilon=0.8),
            paddle.nn.LeakyReLU(negative_slope=0.2),
            paddle.nn.Conv2D(
                in_channels=16,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            paddle.nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.reshape([out.shape[0], 128, self.init_size, self.init_size])
        img = self.conv_blocks(out)
        return img * self.out_scale
