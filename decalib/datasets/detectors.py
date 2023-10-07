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


class FAN(object):
    def __init__(self):
        import face_alignment_paddle

        self.model = face_alignment_paddle.FaceAlignment(
            face_alignment_paddle.LandmarksType._2D, flip_input=False
        )

    def run(self, image):
        """
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        """
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], "kpt68"
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:, (0)])
            right = np.max(kpt[:, (0)])
            top = np.min(kpt[:, (1)])
            bottom = np.max(kpt[:, (1)])
            bbox = [left, top, right, bottom]
            return bbox, "kpt68"


class MTCNN(object):
    def __init__(self, device="cpu"):
        """
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        """
        from facenet_pytorch import MTCNN as mtcnn

        self.place = device
        self.model = mtcnn(keep_all=True)

    def run(self, input):
        """
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box
        """
        out = self.model.detect(input[None, ...])
        if out[0][0] is None:
            return [0]
        else:
            bbox = out[0][0].squeeze()
            return bbox, "bbox"
