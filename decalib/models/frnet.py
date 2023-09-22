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

import math

import cv2
import numpy as np
import paddle


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


class Bottleneck(paddle.nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=1,
            stride=stride,
            bias_attr=False,
        )
        self.bn1 = paddle.nn.BatchNorm2D(num_features=planes)
        self.conv2 = paddle.nn.Conv2D(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
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


class ResNet(paddle.nn.Layer):
    def __init__(self, block, layers, num_classes=1000, include_top=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top
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
        self.maxpool = paddle.nn.MaxPool2D(
            kernel_size=3, stride=2, padding=0, ceil_mode=True
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = paddle.nn.AvgPool2D(kernel_size=7, stride=1, exclusive=False)
        self.fc = paddle.nn.Linear(
            in_features=512 * block.expansion, out_features=num_classes
        )
        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, paddle.nn.BatchNorm2D):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.layer4(x)
        x = self.avgpool(x)
        if not self.include_top:
            return x
        x = x.reshape([x.shape[0], -1])
        x = self.fc(x)
        return x


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


import pickle


def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, "rb") as f:
        weights = pickle.load(f, encoding="latin1")
    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                paddle.assign(paddle.to_tensor(data=param), output=own_state[name])
            except Exception:
                raise RuntimeError(
                    "While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.".format(
                        name, own_state[name].shape, param.shape
                    )
                )
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
