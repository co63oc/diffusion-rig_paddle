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

import numpy as np
import paddle

from . import resnet


class ResnetEncoder(paddle.nn.Layer):
    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder, self).__init__()
        feature_size = 2048
        self.encoder = resnet.load_ResNet50Model()  # out: 2048
        # regressor
        self.layers = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=feature_size, out_features=1024),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=1024, out_features=outsize),
        )
        self.last_op = last_op

    def forward(self, inputs):
        features = self.encoder(inputs)
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters
