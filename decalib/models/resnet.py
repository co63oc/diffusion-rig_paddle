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

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import paddle

import utils.paddle_add

"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
Loads different resnet models
"""
"""
    file:   Resnet.py
    date:   2018_05_02
    author: zhangxiong(1025679612@qq.com)
    mark:   copied from pytorch source code
"""
import math

import numpy as np


class ResNet(paddle.nn.Layer):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias_attr=False,
        )
        self.bn1 = paddle.nn.BatchNorm2D(num_features=64)
        self.relu = paddle.nn.ReLU()
        self.maxpool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = paddle.nn.AvgPool2D(kernel_size=7, stride=1, exclusive=False)

        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                std = math.sqrt(2.0 / n)
                init_normal = paddle.nn.initializer.Normal(mean=0, std=std)
                init_normal(m.weight)
            elif isinstance(m, paddle.nn.BatchNorm2D):
                init_1 = paddle.nn.initializer.Constant(value=1.0)
                init_0 = paddle.nn.initializer.Constant(value=0.0)
                init_1(m.weight)
                init_0(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = paddle.nn.Sequential(
                paddle.nn.Conv2D(
                    in_channels=self.inplanes,
                    out_channels=planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False,
                ),
                paddle.nn.BatchNorm2D(num_features=planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return paddle.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer4(x)
        x2 = self.avgpool(x1)
        x2 = x2.reshape([x2.shape[0], -1])
        return x2


class Bottleneck(paddle.nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=inplanes, out_channels=planes, kernel_size=1, bias_attr=False
        )
        self.bn1 = paddle.nn.BatchNorm2D(num_features=planes)
        self.conv2 = paddle.nn.Conv2D(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias_attr=False,
        )
        self.bn2 = paddle.nn.BatchNorm2D(num_features=planes)
        self.conv3 = paddle.nn.Conv2D(
            in_channels=planes, out_channels=planes * 4, kernel_size=1, bias_attr=False
        )
        self.bn3 = paddle.nn.BatchNorm2D(num_features=planes * 4)
        self.relu = paddle.nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return paddle.nn.Conv2D(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias_attr=False,
    )


class BasicBlock(paddle.nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = paddle.nn.BatchNorm2D(num_features=planes)
        self.relu = paddle.nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = paddle.nn.BatchNorm2D(num_features=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def copy_parameter_from_resnet(model, resnet_dict):
    cur_state_dict = model.state_dict()
    for name, param in list(resnet_dict.items())[0:None]:
        if name not in cur_state_dict:
            continue
        if isinstance(param, paddle.fluid.framework.Parameter):
            param = param.clone().detach()
        try:
            paddle.assign(param, output=cur_state_dict[name])
        except:
            print(name, " is inconsistent!")
            continue


def load_ResNet50Model():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    copy_parameter_from_resnet(
        model, paddle.vision.models.resnet50(pretrained=False).state_dict()
    )
    return model


def load_ResNet101Model():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    copy_parameter_from_resnet(
        model, paddle.vision.models.resnet101(pretrained=True).state_dict()
    )
    return model


def load_ResNet152Model():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    copy_parameter_from_resnet(
        model, paddle.vision.models.resnet152(pretrained=True).state_dict()
    )
    return model


class DoubleConv(paddle.nn.Layer):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            paddle.nn.BatchNorm2D(num_features=out_channels),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            paddle.nn.BatchNorm2D(num_features=out_channels),
            paddle.nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(paddle.nn.Layer):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = paddle.nn.Sequential(
            paddle.nn.MaxPool2D(kernel_size=2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(paddle.nn.Layer):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = paddle.nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
        else:
            self.up = paddle.nn.Conv2DTranspose(
                in_channels=in_channels // 2,
                out_channels=in_channels // 2,
                kernel_size=2,
                stride=2,
            )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = paddle_add.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = paddle.concat(x=[x2, x1], axis=1)
        return self.conv(x)


class OutConv(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)
