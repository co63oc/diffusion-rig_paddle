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
from skimage.io import imread
from skimage.transform import estimate_transform, rescale, resize, warp

from . import detectors


def build_dataloader(config, is_train=True):
    data_list = []
    if "vox1" in config.training_data:
        data_list.append(
            VoxelDataset(
                K=config.K,
                image_size=config.image_size,
                scale=[config.scale_min, config.scale_max],
                n_train=config.n_train,
                isSingle=config.isSingle,
            )
        )
    if "vox2" in config.training_data:
        data_list.append(
            VoxelDataset(
                dataname="vox2",
                K=config.K,
                image_size=config.image_size,
                scale=[config.scale_min, config.scale_max],
                n_train=config.n_train,
                isSingle=config.isSingle,
            )
        )
    if "vggface2" in config.training_data:
        data_list.append(
            VGGFace2Dataset(
                K=config.K,
                image_size=config.image_size,
                scale=[config.scale_min, config.scale_max],
                trans_scale=config.trans_scale,
                isSingle=config.isSingle,
            )
        )
    if "vggface2hq" in config.training_data:
        data_list.append(
            VGGFace2HQDataset(
                K=config.K,
                image_size=config.image_size,
                scale=[config.scale_min, config.scale_max],
                trans_scale=config.trans_scale,
                isSingle=config.isSingle,
            )
        )
    if "ethnicity" in config.training_data:
        data_list.append(
            EthnicityDataset(
                K=config.K,
                image_size=config.image_size,
                scale=[config.scale_min, config.scale_max],
                trans_scale=config.trans_scale,
                isSingle=config.isSingle,
            )
        )
    if "coco" in config.training_data:
        data_list.append(
            COCODataset(
                image_size=config.image_size,
                scale=[config.scale_min, config.scale_max],
                trans_scale=config.trans_scale,
            )
        )
    if "celebahq" in config.training_data:
        data_list.append(
            CelebAHQDataset(
                image_size=config.image_size,
                scale=[config.scale_min, config.scale_max],
                trans_scale=config.trans_scale,
            )
        )
    if "now_eval" in config.training_data:
        data_list.append(NoWVal())
    if "aflw2000" in config.training_data:
        data_list.append(AFLW2000())
    train_dataset = paddle.io.ComposeDataset(data_list)
    if is_train:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    train_loader = paddle.io.ComposeDataset(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    return train_dataset, train_loader


"""
images and keypoints: nomalized to [-1,1]
"""


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


class COCODataset(paddle.io.Dataset):
    def __init__(self, image_size, scale, trans_scale=0, isEval=False):
        """
        # 53877 faces
        K must be less than 6
        """
        self.image_size = image_size
        self.imagefolder = "/ps/scratch/yfeng/Data/COCO/raw/train2017"
        self.kptfolder = "/ps/scratch/yfeng/Data/COCO/face/train2017_kpt"
        self.kptpath_list = os.listdir(self.kptfolder)
        self.scale = scale
        self.trans_scale = trans_scale

    def __len__(self):
        return len(self.kptpath_list)

    def __getitem__(self, idx):
        while 100:
            kptname = self.kptpath_list[idx]
            name = kptname.split("_")[0]
            image_path = os.path.join(self.imagefolder, name + ".jpg")
            kpt_path = os.path.join(self.kptfolder, kptname)
            kpt = np.loadtxt(kpt_path)[:, :2]
            left = np.min(kpt[:, (0)])
            right = np.max(kpt[:, (0)])
            top = np.min(kpt[:, (1)])
            bottom = np.max(kpt[:, (1)])
            if right - left < 10 or bottom - top < 10:
                idx = np.random.randint(low=0, high=len(self.kptpath_list))
                continue
            image = imread(image_path) / 255.0
            if len(image.shape) < 3:
                image = np.tile(image[:, :, (None)], 3)
            tform = self.crop(image, kpt)
            cropped_image = warp(
                image, tform.inverse, output_shape=(self.image_size, self.image_size)
            )
            cropped_kpt = np.dot(
                tform.params, np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T
            ).T
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1
            images_array = paddle.to_tensor(
                data=cropped_image.transpose(2, 0, 1)
            ).astype("float32")
            kpt_array = paddle.to_tensor(data=cropped_kpt).astype("float32")
            data_dict = {"image": images_array * 2.0 - 1, "landmark": kpt_array}
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


class CelebAHQDataset(paddle.io.Dataset):
    def __init__(self, image_size, scale, trans_scale=0, isEval=False):
        """
        # 53877 faces
        K must be less than 6
        """
        self.image_size = image_size
        self.imagefolder = (
            "/ps/project/face2d3d/faceHQ_100K/celebA-HQ/celebahq_resized_256"
        )
        self.kptfolder = (
            "/ps/project/face2d3d/faceHQ_100K/celebA-HQ/celebahq_resized_256_torch"
        )
        self.kptpath_list = os.listdir(self.kptfolder)
        self.scale = scale
        self.trans_scale = trans_scale

    def __len__(self):
        return len(self.kptpath_list)

    def __getitem__(self, idx):
        while 100:
            kptname = self.kptpath_list[idx]
            name = kptname.split(".")[0]
            image_path = os.path.join(self.imagefolder, name + ".png")
            kpt_path = os.path.join(self.kptfolder, kptname)
            kpt = np.load(kpt_path, allow_pickle=True)
            if len(kpt.shape) != 2:
                idx = np.random.randint(low=0, high=len(self.kptpath_list))
                continue
            image = imread(image_path) / 255.0
            if len(image.shape) < 3:
                image = np.tile(image[:, :, (None)], 3)
            tform = self.crop(image, kpt)
            cropped_image = warp(
                image, tform.inverse, output_shape=(self.image_size, self.image_size)
            )
            cropped_kpt = np.dot(
                tform.params, np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T
            ).T
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1
            images_array = paddle.to_tensor(
                data=cropped_image.transpose(2, 0, 1)
            ).astype("float32")
            kpt_array = paddle.to_tensor(data=cropped_kpt).astype("float32")
            data_dict = {"image": images_array, "landmark": kpt_array}
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
        src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])
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


def video2sequence(video_path):
    videofolder = video_path.split(".")[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = video_path.split("/")[-1].split(".")[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = "{}/{}_frame{:04d}.jpg".format(videofolder, video_name, count)
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
        face_detector_model=None,
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
            self.imagepath_list = video2sequence(testpath)
        else:
            print("please check the input path")
            exit()
        print("total {} images".format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        if face_detector == "dlib":
            self.face_detector = detectors.Dlib(model_path=face_detector_model)
        elif face_detector == "fan":
            self.face_detector = detectors.FAN()
        else:
            print("no detector is used")

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = imagepath.split("/")[-1].split(".")[0]
        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:, :, (None)].tile(repeat_times=[1, 1, 3])
        h, w, _ = image.shape
        if self.iscrop:
            if max(h, w) > 1000:
                print("image is too large, resize ", imagepath)
                scale_factor = 1000 / max(h, w)
                image_small = rescale(
                    image, scale_factor, preserve_range=True, multichannel=True
                )
                detected_faces = self.face_detector.run(image_small.astype(np.uint8))
            else:
                detected_faces = self.face_detector.run(image.astype(np.uint8))
            if detected_faces is None:
                print("no face detected! run original image")
                left = 0
                right = h - 1
                top = 0
                bottom = w - 1
            else:
                kpt = detected_faces[0]
                left = np.min(kpt[:, (0)])
                right = np.max(kpt[:, (0)])
                top = np.min(kpt[:, (1)])
                bottom = np.max(kpt[:, (1)])
                if max(h, w) > 1000:
                    scale_factor = 1.0 / scale_factor
                    left = left * scale_factor
                    right = right * scale_factor
                    top = top * scale_factor
                    bottom = bottom * scale_factor
            old_size = (right - left + bottom - top) / 2
            center = np.array(
                [right - (right - left) / 2.0, bottom - (bottom - top) / 2.0]
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
            "tform": tform,
            "original_image": paddle.to_tensor(data=image.transpose(2, 0, 1)).astype(
                dtype="float32"
            ),
        }


class EvalData(paddle.io.Dataset):
    def __init__(
        self,
        testpath,
        kptfolder,
        iscrop=True,
        crop_size=224,
        scale=1.25,
        face_detector="fan",
        face_detector_model=None,
    ):
        """
        testpath: folder, imagepath_list, image path, video path
        """
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath):
            self.imagepath_list = glob(testpath + "/*.jpg") + glob(testpath + "/*.png")
        elif os.path.isfile(testpath) and testpath[-3:] in ["jpg", "png"]:
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and testpath[-3:] in ["mp4", "csv", "vid", "ebm"]:
            self.imagepath_list = video2sequence(testpath)
        else:
            print("please check the input path")
            exit()
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        if face_detector == "dlib":
            self.face_detector = detectors.Dlib(model_path=face_detector_model)
        elif face_detector == "fan":
            self.face_detector = detectors.FAN()
        else:
            print("no detector is used")
        self.kptfolder = kptfolder

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = imagepath.split("/")[-1].split(".")[0]
        image = imread(imagepath)[:, :, :3]
        h, w, _ = image.shape
        if self.iscrop:
            kptpath = os.path.join(self.kptfolder, imagename + ".npy")
            kpt = np.load(kptpath)
            left = np.min(kpt[:, (0)])
            right = np.max(kpt[:, (0)])
            top = np.min(kpt[:, (1)])
            bottom = np.max(kpt[:, (1)])
            old_size = (right - left + bottom - top) / 2
            center = np.array(
                [right - (right - left) / 2.0, bottom - (bottom - top) / 2.0]
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
            "tform": tform,
            "original_image": paddle.to_tensor(data=image.transpose(2, 0, 1)).astype(
                dtype="float32"
            ),
        }
