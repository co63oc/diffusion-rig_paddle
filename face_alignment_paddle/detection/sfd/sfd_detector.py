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
# from paddle.utils.model_zoo import load_url
from paddle.utils.download import get_weights_path_from_url


def load_url(url):
    cache_file = get_weights_path_from_url(url)
    return paddle.load(cache_file)


from ..core import FaceDetector
from .bbox import nms
from .detect import batch_detect, detect
from .net_s3fd import s3fd

models_urls = {
    # 's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
    "s3fd": "https://test1model.oss-cn-beijing.aliyuncs.com/diffusion-rig/paddle_s3fd-619a316812.pth",
}


class SFDDetector(FaceDetector):
    def __init__(self, device, path_to_detector=None, verbose=False):
        super(SFDDetector, self).__init__(device, verbose)

        # Initialise the face detector
        if path_to_detector is None:
            model_weights = load_url(models_urls["s3fd"])
        else:
            model_weights = paddle.load(path_to_detector)

        self.face_detector = s3fd()
        self.face_detector.set_state_dict(model_weights)
        self.face_detector.to(paddle.CUDAPlace(0))
        self.face_detector.eval()

    def _filter_bboxes(self, bboxlist, threshold=0.5):
        if len(bboxlist) > 0:
            keep = nms(bboxlist, 0.3)
            bboxlist = bboxlist[keep, :]
            bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        return bboxlist

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(self.face_detector, image, device=self.device)[0]
        bboxlist = self._filter_bboxes(bboxlist)

        return bboxlist

    def detect_from_batch(self, tensor):
        bboxlists = batch_detect(self.face_detector, tensor, device=self.device)

        new_bboxlists = []
        for i in range(bboxlists.shape[0]):
            bboxlist = bboxlists[i]
            bboxlist = self._filter_bboxes(bboxlist)
            new_bboxlists.append(bboxlist)

        return new_bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
