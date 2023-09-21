import paddle
import numpy as np
from . import resnet


class ResnetEncoder(paddle.nn.Layer):

    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder, self).__init__()
        feature_size = 2048
        self.encoder = resnet.load_ResNet50Model()
        self.layers = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            feature_size, out_features=1024), paddle.nn.ReLU(), paddle.nn.
            Linear(in_features=1024, out_features=outsize))
        self.last_op = last_op

    def forward(self, inputs):
        features = self.encoder(inputs)
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters
