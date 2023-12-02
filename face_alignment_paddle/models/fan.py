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
import paddle.nn as nn
import paddle.nn.functional as F


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    if not bias:
        bias_attr = False
    else:
        bias_attr = None
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=strd,
        padding=padding,
        bias_attr=bias_attr,
    )


class ConvBlock(nn.Layer):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2D(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2D(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2D(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2D(in_planes),
                nn.ReLU(True),
                nn.Conv2D(
                    in_planes, out_planes, kernel_size=1, stride=1, bias_attr=False
                ),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = paddle.concat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Layer):
    def __init__(self, num_sub_layers, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_sub_layers = num_sub_layers
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_sublayer("b1_" + str(level), ConvBlock(self.features, self.features))

        self.add_sublayer("b2_" + str(level), ConvBlock(self.features, self.features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_sublayer(
                "b2_plus_" + str(level), ConvBlock(self.features, self.features)
            )

        self.add_sublayer("b3_" + str(level), ConvBlock(self.features, self.features))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._sub_layers["b1_" + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._sub_layers["b2_" + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._sub_layers["b2_plus_" + str(level)](low2)

        low3 = low2
        low3 = self._sub_layers["b3_" + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode="nearest")

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class FAN(nn.Layer):
    def __init__(self, num_sub_layers=1):
        super(FAN, self).__init__()
        self.num_sub_layers = num_sub_layers

        # Base part
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2D(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_sub_layers):
            self.add_sublayer("m" + str(hg_module), HourGlass(1, 4, 256))
            self.add_sublayer("top_m_" + str(hg_module), ConvBlock(256, 256))
            self.add_sublayer(
                "conv_last" + str(hg_module),
                nn.Conv2D(256, 256, kernel_size=1, stride=1, padding=0),
            )
            self.add_sublayer("bn_end" + str(hg_module), nn.BatchNorm2D(256))
            self.add_sublayer(
                "l" + str(hg_module),
                nn.Conv2D(256, 68, kernel_size=1, stride=1, padding=0),
            )

            if hg_module < self.num_sub_layers - 1:
                self.add_sublayer(
                    "bl" + str(hg_module),
                    nn.Conv2D(256, 256, kernel_size=1, stride=1, padding=0),
                )
                self.add_sublayer(
                    "al" + str(hg_module),
                    nn.Conv2D(68, 256, kernel_size=1, stride=1, padding=0),
                )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_sub_layers):
            hg = self._sub_layers["m" + str(i)](previous)

            ll = hg
            ll = self._sub_layers["top_m_" + str(i)](ll)

            ll = F.relu(
                self._sub_layers["bn_end" + str(i)](
                    self._sub_layers["conv_last" + str(i)](ll)
                ),
                True,
            )

            # Predict heatmaps
            tmp_out = self._sub_layers["l" + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_sub_layers - 1:
                ll = self._sub_layers["bl" + str(i)](ll)
                tmp_out_ = self._sub_layers["al" + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs
