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

import warnings

import cv2
import dlib

from ...utils import load_file_from_url
from ..core import FaceDetector


class DlibDetector(FaceDetector):
    def __init__(self, device, path_to_detector=None, verbose=False):
        super().__init__(device, verbose)

        warnings.warn(
            "Warning: this detector is deprecated. Please use a different one, i.e.: S3FD."
        )

        # Initialise the face detector
        if "cuda" in device:
            if path_to_detector is None:
                path_to_detector = load_file_from_url(
                    "https://www.adrianbulat.com/downloads/dlib/mmod_human_face_detector.dat"
                )

            self.face_detector = dlib.cnn_face_detection_model_v1(path_to_detector)
        else:
            self.face_detector = dlib.get_frontal_face_detector()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path, rgb=False)

        detected_faces = self.face_detector(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        if "cuda" not in self.device:
            detected_faces = [
                [d.left(), d.top(), d.right(), d.bottom()] for d in detected_faces
            ]
        else:
            detected_faces = [
                [d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()]
                for d in detected_faces
            ]

        return detected_faces

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
