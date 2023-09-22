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

import cv2
import numpy as np
import paddle
import scipy
import scipy.io
from skimage.io import imread
from skimage.transform import estimate_transform, resize, warp

from . import detectors


def video2sequence(video_path, sample_step=10):
    videofolder = os.path.splitext(video_path)[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = os.path.join(videofolder, f"{video_name}_frame{count:04d}.jpg")
        cv2.imwrite(imagepath, image)
        success, image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print("video frames are stored in {}".format(videofolder))
    return imagepath_list


class TestData(paddle.io.Dataset):
    def __init__(
        self,
        testpath,
        iscrop=True,
        crop_size=224,
        scale=1.25,
        face_detector="fan",
        sample_step=10,
        size=256,
        sort=False,
    ):
        """
        testpath: folder, imagepath_list, image path, video path
        """
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath):
            self.imagepath_list = (
                glob(testpath + "/*.jpg")
                + glob(testpath + "/*.png")
                + glob(testpath + "/*.bmp")
            )
        elif os.path.isfile(testpath) and testpath[-3:] in ["jpg", "png", "bmp"]:
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and testpath[-3:] in ["mp4", "csv", "vid", "ebm"]:
            self.imagepath_list = video2sequence(testpath, sample_step)
        else:
            print(f"please check the test path: {testpath}")
            exit()
        if sort:
            self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        self.size = size
        if face_detector == "fan":
            self.face_detector = detectors.FAN()
        else:
            print(f"please check the detector: {face_detector}")
            exit()

    def __len__(self):
        return len(self.imagepath_list)

    def bbox2point(self, left, right, top, bottom, type="bbox"):
        """bbox from detector and landmarks are different"""
        if type == "kpt68":
            old_size = (right - left + bottom - top) / 2 * 1.1
            center = np.array(
                [right - (right - left) / 2.0, bottom - (bottom - top) / 2.0]
            )
        elif type == "bbox":
            old_size = (right - left + bottom - top) / 2
            center = np.array(
                [
                    right - (right - left) / 2.0,
                    bottom - (bottom - top) / 2.0 + old_size * 0.12,
                ]
            )
        else:
            raise NotImplementedError
        return old_size, center

    def get_image(self, image):
        h, w, _ = image.shape
        bbox, bbox_type = self.face_detector.run(image)
        if len(bbox) < 4:
            print("no face detected! run original image")
            left = 0
            right = h - 1
            top = 0
            bottom = w - 1
        else:
            left = bbox[0]
            right = bbox[2]
            top = bbox[1]
            bottom = bbox[3]
        old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
        size = int(old_size * self.scale)
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
            "tform": paddle.to_tensor(data=tform.params).astype(dtype="float32"),
            "original_image": paddle.to_tensor(data=image.transpose(2, 0, 1)).astype(
                dtype="float32"
            ),
        }

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        im = imread(imagepath)
        if self.size is not None:
            im = (
                resize(im, (self.size, self.size), anti_aliasing=True) * 255.0
            ).astype(np.uint8)
        image = np.array(im)
        if len(image.shape) == 2:
            image = image[:, :, (None)].tile(repeat_times=[1, 1, 3])
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        h, w, _ = image.shape
        if self.iscrop:
            kpt_matpath = os.path.splitext(imagepath)[0] + ".mat"
            kpt_txtpath = os.path.splitext(imagepath)[0] + ".txt"
            if os.path.exists(kpt_matpath):
                kpt = scipy.io.loadmat(kpt_matpath)["pt3d_68"].T
                left = np.min(kpt[:, (0)])
                right = np.max(kpt[:, (0)])
                top = np.min(kpt[:, (1)])
                bottom = np.max(kpt[:, (1)])
                old_size, center = self.bbox2point(
                    left, right, top, bottom, type="kpt68"
                )
            elif os.path.exists(kpt_txtpath):
                kpt = np.loadtxt(kpt_txtpath)
                left = np.min(kpt[:, (0)])
                right = np.max(kpt[:, (0)])
                top = np.min(kpt[:, (1)])
                bottom = np.max(kpt[:, (1)])
                old_size, center = self.bbox2point(
                    left, right, top, bottom, type="kpt68"
                )
            else:
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 4:
                    print("no face detected! run original image")
                    left = 0
                    right = h - 1
                    top = 0
                    bottom = w - 1
                else:
                    left = bbox[0]
                    right = bbox[2]
                    top = bbox[1]
                    bottom = bbox[3]
                old_size, center = self.bbox2point(
                    left, right, top, bottom, type=bbox_type
                )
            size = int(old_size * self.scale)
            src_pts = np.array(
                [
                    [center[0] - size / 2, center[1] - size / 2],
                    [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2],
                ]
            )
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])
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
            "tform": paddle.to_tensor(data=tform.params).astype(dtype="float32"),
            "original_image": paddle.to_tensor(data=image.transpose(2, 0, 1)).astype(
                dtype="float32"
            ),
        }
