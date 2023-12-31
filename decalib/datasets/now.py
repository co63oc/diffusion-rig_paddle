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
from glob import glob

import numpy as np
import paddle
from skimage.io import imread
from skimage.transform import estimate_transform, resize, warp


class NoWDataset(paddle.io.Dataset):
    def __init__(self, ring_elements=6, crop_size=224, scale=1.6):
        folder = "/ps/scratch/yfeng/other-github/now_evaluation/data/NoW_Dataset"
        self.data_path = os.path.join(folder, "imagepathsvalidation.txt")
        with open(self.data_path) as f:
            self.data_lines = f.readlines()
        self.imagefolder = os.path.join(
            folder, "final_release_version", "iphone_pictures"
        )
        self.bbxfolder = os.path.join(folder, "final_release_version", "detected_face")
        self.crop_size = crop_size
        self.scale = scale

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, index):
        imagepath = os.path.join(self.imagefolder, self.data_lines[index].strip())
        bbx_path = os.path.join(
            self.bbxfolder, self.data_lines[index].strip().replace(".jpg", ".npy")
        )
        bbx_data = np.load(bbx_path, allow_pickle=True, encoding="latin1").item()
        left = bbx_data["left"]
        right = bbx_data["right"]
        top = bbx_data["top"]
        bottom = bbx_data["bottom"]
        imagename = imagepath.split("/")[-1].split(".")[0]
        image = imread(imagepath)[:, :, :3]
        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * self.scale)
        src_pts = np.array(
            [
                [center[0] - size / 2, center[1] - size / 2],
                [center[0] - size / 2, center[1] + size / 2],
                [center[0] + size / 2, center[1] - size / 2],
            ]
        )
        DST_PTS = np.array([[0, 0], [0, self.crop_size - 1], [self.crop_size - 1, 0]])
        tform = estimate_transform("similarity", src_pts, DST_PTS)
        image = image / 255.0
        dst_image = warp(
            image, tform.inverse, output_shape=(self.crop_size, self.crop_size)
        )
        dst_image = dst_image.transpose(2, 0, 1)
        return {
            "image": paddle.to_tensor(data=dst_image).astype(dtype="float32"),
            "imagename": self.data_lines[index].strip().replace(".jpg", ""),
        }
