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
import sys
from glob import glob

import cv2
import numpy as np
import paddle
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, rescale, resize, warp


class VoxelDataset(paddle.io.Dataset):
    def __init__(
        self,
        K,
        image_size,
        scale,
        trans_scale=0,
        dataname="vox2",
        n_train=100000,
        isTemporal=False,
        isEval=False,
        isSingle=False,
    ):
        self.K = K
        self.image_size = image_size
        if dataname == "vox1":
            self.kpt_suffix = ".txt"
            self.imagefolder = "/ps/project/face2d3d/VoxCeleb/vox1/dev/images_cropped"
            self.kptfolder = "/ps/scratch/yfeng/Data/VoxCeleb/vox1/landmark_2d"
            self.face_dict = {}
            for person_id in sorted(os.listdir(self.kptfolder)):
                for video_id in os.listdir(os.path.join(self.kptfolder, person_id)):
                    for face_id in os.listdir(
                        os.path.join(self.kptfolder, person_id, video_id)
                    ):
                        if "txt" in face_id:
                            continue
                        key = person_id + "/" + video_id + "/" + face_id
                        name_list = os.listdir(
                            os.path.join(self.kptfolder, person_id, video_id, face_id)
                        )
                        name_list = [name.split["."][0] for name in name_list]
                        if len(name_list) < self.K:
                            continue
                        self.face_dict[key] = sorted(name_list)
        elif dataname == "vox2":
            self.kpt_suffix = ".npy"
            self.imagefolder = (
                "/ps/scratch/face2d3d/VoxCeleb/vox2/dev/images_cropped_full_height"
            )
            self.kptfolder = "/ps/scratch/face2d3d/vox2_best_clips_annotated_torch7"
            self.segfolder = "/ps/scratch/face2d3d/texture_in_the_wild_code/vox2_best_clips_cropped_frames_seg/test_crop_size_400_batch/"
            cleanlist_path = "/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/vox2_best_clips_info_max_normal_50_images_loadinglist.npy"
            cleanlist = np.load(cleanlist_path, allow_pickle=True)
            self.face_dict = {}
            for line in cleanlist:
                person_id, video_id, face_id, name = line.split("/")
                key = person_id + "/" + video_id + "/" + face_id
                if key not in self.face_dict.keys():
                    self.face_dict[key] = []
                else:
                    self.face_dict[key].append(name)
            keys = list(self.face_dict.keys())
            for key in keys:
                if len(self.face_dict[key]) < self.K:
                    del self.face_dict[key]
        self.face_list = list(self.face_dict.keys())
        n_train = n_train if n_train < len(self.face_list) else len(self.face_list)
        self.face_list = list(self.face_dict.keys())[:n_train]
        if isEval:
            self.face_list = list(self.face_dict.keys())[:n_train][-100:]
        self.isTemporal = isTemporal
        self.scale = scale
        self.trans_scale = trans_scale
        self.isSingle = isSingle
        if isSingle:
            self.K = 1

    def __len__(self):
        return len(self.face_list)

    def __getitem__(self, idx):
        key = self.face_list[idx]
        person_id, video_id, face_id = key.split("/")
        name_list = self.face_dict[key]
        ind = np.random.randint(low=0, high=len(name_list))
        images_list = []
        kpt_list = []
        fullname_list = []
        mask_list = []
        if self.isTemporal:
            random_start = np.random.randint(low=0, high=len(name_list) - self.K)
            sample_list = range(random_start, random_start + self.K)
        else:
            sample_list = np.array(
                np.random.randint(low=0, high=len(name_list), size=self.K)
            )
        for i in sample_list:
            name = name_list[i]
            image_path = os.path.join(
                self.imagefolder, person_id, video_id, face_id, name + ".png"
            )
            kpt_path = os.path.join(
                self.kptfolder, person_id, video_id, face_id, name + self.kpt_suffix
            )
            seg_path = os.path.join(
                self.segfolder, person_id, video_id, face_id, name + ".npy"
            )
            image = imread(image_path) / 255.0
            kpt = np.load(kpt_path)[:, :2]
            mask = self.load_mask(seg_path, image.shape[0], image.shape[1])
            tform = self.crop(image, kpt)
            cropped_image = warp(
                image, tform.inverse, output_shape=(self.image_size, self.image_size)
            )
            cropped_mask = warp(
                mask, tform.inverse, output_shape=(self.image_size, self.image_size)
            )
            cropped_kpt = np.dot(
                tform.params, np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T
            ).T
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1
            images_list.append(cropped_image.transpose(2, 0, 1))
            kpt_list.append(cropped_kpt)
            mask_list.append(cropped_mask)
        images_array = paddle.to_tensor(data=np.array(images_list)).astype("float32")
        kpt_array = paddle.to_tensor(data=np.array(kpt_list)).astype("float32")
        mask_array = paddle.to_tensor(data=np.array(mask_list)).astype("float32")
        if self.isSingle:
            images_array = images_array.squeeze()
            kpt_array = kpt_array.squeeze()
            mask_array = mask_array.squeeze()
        data_dict = {"image": images_array, "landmark": kpt_array, "mask": mask_array}
        return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:, (0)])
        right = np.max(kpt[:, (0)])
        top = np.min(kpt[:, (1)])
        bottom = np.max(kpt[:, (1)])
        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size
        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size * scale)
        src_pts = np.array(
            [
                [center[0] - size / 2, center[1] - size / 2],
                [center[0] - size / 2, center[1] + size / 2],
                [center[0] + size / 2, center[1] - size / 2],
            ]
        )
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform("similarity", src_pts, DST_PTS)
        return tform

    def load_mask(self, maskpath, h, w):
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            mask = np.zeros_like(vis_parsing_anno)
            mask[vis_parsing_anno > 0.5] = 1.0
        else:
            mask = np.ones((h, w))
        return mask
