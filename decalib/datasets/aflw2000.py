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
import scipy
import scipy.io
from skimage.io import imread
from skimage.transform import estimate_transform, warp


class AFLW2000(paddle.io.Dataset):
    def __init__(self, testpath="/ps/scratch/yfeng/Data/AFLW2000/GT", crop_size=224):
        """
        data class for loading AFLW2000 dataset
        make sure each image has corresponding mat file, which provides cropping information
        """
        if os.path.isdir(testpath):
            self.imagepath_list = glob(testpath + "/*.jpg") + glob(testpath + "/*.png")
        elif isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isfile(testpath) and testpath[-3:] in ["jpg", "png"]:
            self.imagepath_list = [testpath]
        else:
            print("please check the input path")
            exit()
        print("total {} images".format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = 1.6
        self.resolution_inp = crop_size

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = imagepath.split("/")[-1].split(".")[0]
        image = imread(imagepath)[:, :, :3]
        kpt = scipy.io.loadmat(imagepath.replace("jpg", "mat"))["pt3d_68"].T
        left = np.min(kpt[:, (0)])
        right = np.max(kpt[:, (0)])
        top = np.min(kpt[:, (1)])
        bottom = np.max(kpt[:, (1)])
        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * self.scale)

        # crop image
        src_pts = np.array(
            [
                [center[0] - size / 2, center[1] - size / 2],
                [center[0] - size / 2, center[1] + size / 2],
                [center[0] + size / 2, center[1] - size / 2],
            ]
        )
        DST_PTS = np.array(
            [[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]]
        )
        tform = estimate_transform("similarity", src_pts, DST_PTS)
        image = image / 255.0
        dst_image = warp(
            image,
            tform.inverse,
            output_shape=(self.resolution_inp, self.resolution_inp),
        )
        dst_image = dst_image.transpose(2, 0, 1)
        return {
            "image": paddle.to_tensor(data=dst_image).astype(dtype="float32"),
            "imagename": imagename,
        }
