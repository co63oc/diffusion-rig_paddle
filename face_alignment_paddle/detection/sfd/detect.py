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

import argparse
import datetime
import math
import os
import random
import sys
import zipfile

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
import scipy.io as sio

from .bbox import *
from .net_s3fd import s3fd


def detect(net, img, device):
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)
    # Creates a batch of 1
    img = img.reshape((1,) + img.shape)

    img = paddle.to_tensor(img, place=paddle.CUDAPlace(0)).astype("float32")

    return batch_detect(net, img, device)


def batch_detect(net, img_batch, device):
    """
    Inputs:
        - img_batch: a paddle.Tensor of shape (Batch size, Channels, Height, Width)
    """

    # if 'cuda' in device:
    #     paddle.backends.cudnn.benchmark = True

    BB, CC, HH, WW = img_batch.shape

    with paddle.no_grad():
        olist = net(img_batch.astype("float32"))  # patched uint8_t overflow error

    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], axis=1)

    bboxlists = []

    olist = [oelem.cpu() for oelem in olist]

    for j in range(BB):
        bboxlist = []
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            FB, FC, FH, FW = ocls.shape  # feature map size
            stride = 2 ** (i + 2)  # 4,8,16,32,64,128
            anchor = stride * 4
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for Iindex, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[j, 1, hindex, windex]
                loc = oreg[j, :, hindex, windex].reshape((1, 4))
                priors = paddle.to_tensor(
                    [[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]
                )
                variances = [0.1, 0.2]
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                bboxlist.append([x1, y1, x2, y2, score])

        bboxlists.append(bboxlist)

    bboxlists = np.array(bboxlists)
    return bboxlists


def flip_detect(net, img, device):
    img = cv2.flip(img, 1)
    b = detect(net, img, device)

    bboxlist = np.zeros(b.shape)
    bboxlist[:, 0] = img.shape[1] - b[:, 2]
    bboxlist[:, 1] = b[:, 1]
    bboxlist[:, 2] = img.shape[1] - b[:, 0]
    bboxlist[:, 3] = b[:, 3]
    bboxlist[:, 4] = b[:, 4]
    return bboxlist


def pts_to_bb(pts):
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    return np.array([min_x, min_y, max_x, max_y])
