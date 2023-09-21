import paddle
import os, sys
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob


class VGGFace2Dataset(paddle.io.Dataset):

    def __init__(self, K, image_size, scale, trans_scale=0, isTemporal=
        False, isEval=False, isSingle=False):
        """
        K must be less than 6
        """
        self.K = K
        self.image_size = image_size
        self.imagefolder = '/ps/scratch/face2d3d/train'
        self.kptfolder = '/ps/scratch/face2d3d/train_annotated_torch7'
        self.segfolder = (
            '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_seg/test_crop_size_400_batch'
            )
        datafile = (
            '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_train_list_max_normal_100_ring_5_1_serial.npy'
            )
        if isEval:
            datafile = (
                '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_val_list_max_normal_100_ring_5_1_serial.npy'
                )
        self.data_lines = np.load(datafile).astype('str')
        self.isTemporal = isTemporal
        self.scale = scale
        self.trans_scale = trans_scale
        self.isSingle = isSingle
        if isSingle:
            self.K = 1

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        images_list = []
        kpt_list = []
        mask_list = []
        random_ind = np.random.permutation(5)[:self.K]
        for i in random_ind:
            name = self.data_lines[idx, i]
            image_path = os.path.join(self.imagefolder, name + '.jpg')
            seg_path = os.path.join(self.segfolder, name + '.npy')
            kpt_path = os.path.join(self.kptfolder, name + '.npy')
            image = imread(image_path) / 255.0
            kpt = np.load(kpt_path)[:, :2]
            mask = self.load_mask(seg_path, image.shape[0], image.shape[1])
            tform = self.crop(image, kpt)
            cropped_image = warp(image, tform.inverse, output_shape=(self.
                image_size, self.image_size))
            cropped_mask = warp(mask, tform.inverse, output_shape=(self.
                image_size, self.image_size))
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt
                .shape[0], 1])]).T).T
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1
            images_list.append(cropped_image.transpose(2, 0, 1))
            kpt_list.append(cropped_kpt)
            mask_list.append(cropped_mask)
        images_array = paddle.to_tensor(data=np.array(images_list)).astype(
            'float32')
        kpt_array = paddle.to_tensor(data=np.array(kpt_list)).astype('float32')
        mask_array = paddle.to_tensor(data=np.array(mask_list)).astype(
            'float32')
        if self.isSingle:
            images_array = images_array.squeeze()
            kpt_array = kpt_array.squeeze()
            mask_array = mask_array.squeeze()
        data_dict = {'image': images_array, 'landmark': kpt_array, 'mask':
            mask_array}
        return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:, (0)])
        right = np.max(kpt[:, (0)])
        top = np.min(kpt[:, (1)])
        bottom = np.max(kpt[:, (1)])
        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom -
            top) / 2.0])
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size
        scale = np.random.rand() * (self.scale[1] - self.scale[0]
            ) + self.scale[0]
        size = int(old_size * scale)
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [
            center[0] - size / 2, center[1] + size / 2], [center[0] + size /
            2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.
            image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        return tform

    def load_mask(self, maskpath, h, w):
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            mask = np.zeros_like(vis_parsing_anno)
            mask[vis_parsing_anno > 0.5] = 1.0
        else:
            mask = np.ones((h, w))
        return mask


class VGGFace2HQDataset(paddle.io.Dataset):

    def __init__(self, K, image_size, scale, trans_scale=0, isTemporal=
        False, isEval=False, isSingle=False):
        """
        K must be less than 6
        """
        self.K = K
        self.image_size = image_size
        self.imagefolder = '/ps/scratch/face2d3d/train'
        self.kptfolder = '/ps/scratch/face2d3d/train_annotated_torch7'
        self.segfolder = (
            '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_seg/test_crop_size_400_batch'
            )
        datafile = (
            '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_bbx_size_bigger_than_400_train_list_max_normal_100_ring_5_1_serial.npy'
            )
        self.data_lines = np.load(datafile).astype('str')
        self.isTemporal = isTemporal
        self.scale = scale
        self.trans_scale = trans_scale
        self.isSingle = isSingle
        if isSingle:
            self.K = 1

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        images_list = []
        kpt_list = []
        mask_list = []
        for i in range(self.K):
            name = self.data_lines[idx, i]
            image_path = os.path.join(self.imagefolder, name + '.jpg')
            seg_path = os.path.join(self.segfolder, name + '.npy')
            kpt_path = os.path.join(self.kptfolder, name + '.npy')
            image = imread(image_path) / 255.0
            kpt = np.load(kpt_path)[:, :2]
            mask = self.load_mask(seg_path, image.shape[0], image.shape[1])
            tform = self.crop(image, kpt)
            cropped_image = warp(image, tform.inverse, output_shape=(self.
                image_size, self.image_size))
            cropped_mask = warp(mask, tform.inverse, output_shape=(self.
                image_size, self.image_size))
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt
                .shape[0], 1])]).T).T
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1
            images_list.append(cropped_image.transpose(2, 0, 1))
            kpt_list.append(cropped_kpt)
            mask_list.append(cropped_mask)
        images_array = paddle.to_tensor(data=np.array(images_list)).astype(
            'float32')
        kpt_array = paddle.to_tensor(data=np.array(kpt_list)).astype('float32')
        mask_array = paddle.to_tensor(data=np.array(mask_list)).astype(
            'float32')
        if self.isSingle:
            images_array = images_array.squeeze()
            kpt_array = kpt_array.squeeze()
            mask_array = mask_array.squeeze()
        data_dict = {'image': images_array, 'landmark': kpt_array, 'mask':
            mask_array}
        return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:, (0)])
        right = np.max(kpt[:, (0)])
        top = np.min(kpt[:, (1)])
        bottom = np.max(kpt[:, (1)])
        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom -
            top) / 2.0])
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size
        scale = np.random.rand() * (self.scale[1] - self.scale[0]
            ) + self.scale[0]
        size = int(old_size * scale)
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [
            center[0] - size / 2, center[1] + size / 2], [center[0] + size /
            2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.
            image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        return tform

    def load_mask(self, maskpath, h, w):
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            mask = np.zeros_like(vis_parsing_anno)
            mask[vis_parsing_anno > 0.5] = 1.0
        else:
            mask = np.ones((h, w))
        return mask
