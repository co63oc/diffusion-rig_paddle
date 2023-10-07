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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from functools import reduce

import numpy as np
import paddle
import torchfile

from utils import paddle_add


def l2_distance(verts1, verts2):
    return paddle.sqrt(x=((verts1 - verts2) ** 2).sum(axis=2)).mean(axis=1).mean()


# VAE
def kl_loss(texcode):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mu, logvar = texcode[:, :128], texcode[:, 128:]
    KLD_element = (
        mu.pow(y=2)
        .add_(y=paddle.to_tensor(logvar.exp()))
        .scale_(-1)
        .add_(y=paddle.to_tensor(1))
        .add_(y=paddle.to_tensor(logvar))
    )
    KLD = paddle.sum(x=KLD_element).scale_(-0.5)
    # KL divergence
    return KLD


# ------------------------------------- Losses/Regularizations for shading
# white shading
# uv_mask_tf = tf.expand_dims(tf.expand_dims(tf.constant( self.uv_mask, dtype = tf.float32 ), 0), -1)
# mean_shade = tf.reduce_mean( tf.multiply(shade_300W, uv_mask_tf) , axis=[0,1,2]) * 16384 / 10379
# G_loss_white_shading = 10*norm_loss(mean_shade,  0.99*tf.ones([1, 3], dtype=tf.float32), loss_type = "l2")
def shading_white_loss(shading):
    """
    regularize lighting: assume lights close to white
    """
    rgb_diff = (shading.mean(axis=[0, 2, 3]) - 0.99) ** 2
    return rgb_diff.mean()


def shading_smooth_loss(shading):
    """
    assume: shading should be smooth
    ref: Lifting AutoEncoders: Unsupervised Learning of a Fully-Disentangled 3D Morphable Model using Deep Non-Rigid Structure from Motion
    """
    dx = shading[:, :, 1:-1, 1:] - shading[:, :, 1:-1, :-1]
    dy = shading[:, :, 1:, 1:-1] - shading[:, :, :-1, 1:-1]
    gradient_image = (dx**2).mean() + (dy**2).mean()
    return gradient_image.mean()


# ------------------------------------- Losses/Regularizations for albedo
def albedo_constancy_loss(albedo, alpha=15, weight=1.0):
    """
    for similarity of neighbors
    ref: Self-supervised Multi-level Face Model Learning for Monocular Reconstruction at over 250 Hz
        Towards High-fidelity Nonlinear 3D Face Morphable Model
    """
    albedo_chromaticity = albedo / (paddle.sum(x=albedo, axis=1, keepdim=True) + 1e-06)
    weight_x = paddle.exp(
        x=-alpha
        * (albedo_chromaticity[:, :, 1:, :] - albedo_chromaticity[:, :, :-1, :]) ** 2
    ).detach()
    weight_y = paddle.exp(
        x=-alpha
        * (albedo_chromaticity[:, :, :, 1:] - albedo_chromaticity[:, :, :, :-1]) ** 2
    ).detach()
    albedo_const_loss_x = (albedo[:, :, 1:, :] - albedo[:, :, :-1, :]) ** 2 * weight_x
    albedo_const_loss_y = (albedo[:, :, :, 1:] - albedo[:, :, :, :-1]) ** 2 * weight_y
    albedo_constancy_loss = albedo_const_loss_x.mean() + albedo_const_loss_y.mean()
    return albedo_constancy_loss * weight


def albedo_ring_loss(texcode, ring_elements, margin, weight=1.0):
    """
    computes ring loss for ring_outputs before FLAME decoder
    Inputs:
      ring_outputs = a list containing N streams of the ring; len(ring_outputs) = N
      Each ring_outputs[i] is a tensor of (batch_size X shape_dim_num)
      Aim is to force each row (same subject) of each stream to produce same shape
      Each row of first N-1 strams are of the same subject and
      the Nth stream is the different subject
    """
    tot_ring_loss = (texcode[0] - texcode[0]).sum()
    diff_stream = texcode[-1]
    count = 0.0
    for i in range(ring_elements - 1):
        for j in range(ring_elements - 1):
            pd = (texcode[i] - texcode[j]).pow(y=2).sum(axis=1)
            nd = (texcode[i] - diff_stream).pow(y=2).sum(axis=1)
            tot_ring_loss = paddle.add(
                x=tot_ring_loss,
                y=paddle.to_tensor(
                    paddle.nn.functional.relu(x=margin + pd - nd).mean()
                ),
            )
            count += 1.0
    tot_ring_loss = 1.0 / count * tot_ring_loss
    return tot_ring_loss * weight


def albedo_same_loss(albedo, ring_elements, weight=1.0):
    """
    computes ring loss for ring_outputs before FLAME decoder
    Inputs:
      ring_outputs = a list containing N streams of the ring; len(ring_outputs) = N
      Each ring_outputs[i] is a tensor of (batch_size X shape_dim_num)
      Aim is to force each row (same subject) of each stream to produce same shape
      Each row of first N-1 strams are of the same subject and
      the Nth stream is the different subject
    """
    loss = 0
    for i in range(ring_elements - 1):
        for j in range(ring_elements - 1):
            pd = (albedo[i] - albedo[j]).pow(y=2).mean()
            loss += pd
    loss = loss / ring_elements
    return loss * weight


def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, weights=None):
    """
    Computes the l1 loss between the ground truth keypoints and the predicted keypoints
    Inputs:
    kp_gt  : N x K x 3
    kp_pred: N x K x 2
    """
    if weights is not None:
        real_2d_kp[:, :, (2)] = weights[(None), :] * real_2d_kp[:, :, (2)]
    kp_gt = real_2d_kp.reshape([-1, 3])
    kp_pred = predicted_2d_kp.reshape([-1, 2])
    vis = kp_gt[:, (2)]
    k = paddle.sum(x=vis) * 2.0 + 1e-08
    dif_abs = paddle.abs(x=kp_gt[:, :2] - kp_pred).sum(axis=1)
    return paddle.matmul(x=dif_abs, y=vis) * 1.0 / k


def landmark_loss(predicted_landmarks, landmarks_gt, weight=1.0):
    if paddle.is_tensor(x=landmarks_gt) is not True:
        real_2d = paddle.concat(x=landmarks_gt)
    else:
        real_2d = paddle.concat(
            x=[landmarks_gt, paddle.ones(shape=(landmarks_gt.shape[0], 68, 1))], axis=-1
        )
    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks)
    return loss_lmk_2d * weight


def eye_dis(landmarks):
    eye_up = landmarks[:, ([37, 38, 43, 44]), :]
    eye_bottom = landmarks[:, ([41, 40, 47, 46]), :]
    dis = paddle.sqrt(x=((eye_up - eye_bottom) ** 2).sum(axis=2))
    return dis


def eyed_loss(predicted_landmarks, landmarks_gt, weight=1.0):
    if paddle.is_tensor(x=landmarks_gt) is not True:
        real_2d = paddle.concat(x=landmarks_gt)
    else:
        real_2d = paddle.concat(
            x=[landmarks_gt, paddle.ones(shape=(landmarks_gt.shape[0], 68, 1))], axis=-1
        )
    pred_eyed = eye_dis(predicted_landmarks[:, :, :2])
    gt_eyed = eye_dis(real_2d[:, :, :2])
    loss = (pred_eyed - gt_eyed).abs().mean()
    return loss


def lip_dis(landmarks):
    lip_up = landmarks[:, ([61, 62, 63]), :]
    lip_down = landmarks[:, ([67, 66, 65]), :]
    dis = paddle.sqrt(x=((lip_up - lip_down) ** 2).sum(axis=2))
    return dis


def lipd_loss(predicted_landmarks, landmarks_gt, weight=1.0):
    if paddle.is_tensor(x=landmarks_gt) is not True:
        real_2d = paddle.concat(x=landmarks_gt)
    else:
        real_2d = paddle.concat(
            x=[landmarks_gt, paddle.ones(shape=(landmarks_gt.shape[0], 68, 1))], axis=-1
        )
    pred_lipd = lip_dis(predicted_landmarks[:, :, :2])
    gt_lipd = lip_dis(real_2d[:, :, :2])
    loss = (pred_lipd - gt_lipd).abs().mean()
    return loss


def weighted_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.0):
    # smaller inner landmark weights
    real_2d = landmarks_gt
    weights = paddle.ones(shape=(68,))
    weights[5:7] = 2
    weights[10:12] = 2
    # nose points
    weights[27:36] = 1.5
    weights[30] = 3
    weights[31] = 3
    weights[35] = 3
    # inner mouth
    weights[60:68] = 1.5
    weights[48:60] = 1.5
    weights[48] = 3
    weights[54] = 3

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
    return loss_lmk_2d * weight


def landmark_loss_tensor(predicted_landmarks, landmarks_gt, weight=1.0):
    loss_lmk_2d = batch_kp_2d_l1_loss(landmarks_gt, predicted_landmarks)
    return loss_lmk_2d * weight


def ring_loss(ring_outputs, ring_type, margin, weight=1.0):
    """
    computes ring loss for ring_outputs before FLAME decoder
    Inputs:
        ring_outputs = a list containing N streams of the ring; len(ring_outputs) = N
        Each ring_outputs[i] is a tensor of (batch_size X shape_dim_num)
        Aim is to force each row (same subject) of each stream to produce same shape
        Each row of first N-1 strams are of the same subject and
        the Nth stream is the different subject
    """
    tot_ring_loss = (ring_outputs[0] - ring_outputs[0]).sum()
    if ring_type == "51":
        diff_stream = ring_outputs[-1]
        count = 0.0
        for i in range(6):
            for j in range(6):
                pd = (ring_outputs[i] - ring_outputs[j]).pow(y=2).sum(axis=1)
                nd = (ring_outputs[i] - diff_stream).pow(y=2).sum(axis=1)
                tot_ring_loss = paddle.add(
                    x=tot_ring_loss,
                    y=paddle.to_tensor(
                        paddle.nn.functional.relu(x=margin + pd - nd).mean()
                    ),
                )
                count += 1.0
    elif ring_type == "33":
        perm_code = [
            (0, 1, 3),
            (0, 1, 4),
            (0, 1, 5),
            (0, 2, 3),
            (0, 2, 4),
            (0, 2, 5),
            (1, 0, 3),
            (1, 0, 4),
            (1, 0, 5),
            (1, 2, 3),
            (1, 2, 4),
            (1, 2, 5),
            (2, 0, 3),
            (2, 0, 4),
            (2, 0, 5),
            (2, 1, 3),
            (2, 1, 4),
            (2, 1, 5),
        ]
        count = 0.0
        for i in perm_code:
            pd = (ring_outputs[i[0]] - ring_outputs[i[1]]).pow(y=2).sum(axis=1)
            nd = (ring_outputs[i[1]] - ring_outputs[i[2]]).pow(y=2).sum(axis=1)
            tot_ring_loss = paddle.add(
                x=tot_ring_loss,
                y=paddle.to_tensor(
                    paddle.nn.functional.relu(x=margin + pd - nd).mean()
                ),
            )
            count += 1.0
    tot_ring_loss = 1.0 / count * tot_ring_loss
    return tot_ring_loss * weight


# images/features/perceptual
def gradient_dif_loss(prediction, gt):
    prediction_diff_x = prediction[:, :, 1:-1, 1:] - prediction[:, :, 1:-1, :-1]
    prediction_diff_y = prediction[:, :, 1:, 1:-1] - prediction[:, :, 1:, 1:-1]
    gt_x = gt[:, :, 1:-1, 1:] - gt[:, :, 1:-1, :-1]
    gt_y = gt[:, :, 1:, 1:-1] - gt[:, :, :-1, 1:-1]
    diff = paddle.mean(x=(prediction_diff_x - gt_x) ** 2) + paddle.mean(
        x=(prediction_diff_y - gt_y) ** 2
    )
    return diff.mean()


def get_laplacian_kernel2d(kernel_size: int):
    """Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (int): filter size should be odd.

    Returns:
        Tensor: 2D tensor with laplacian filter matrix coefficients.

    Shape:
        - Output: :math:`(\\text{kernel_size}_x, \\text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])

        >>> kornia.image.get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])

    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size <= 0:
        raise TypeError(
            "ksize must be an odd positive integer. Got {}".format(kernel_size)
        )
    kernel = paddle.ones(shape=(kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size**2
    kernel_2d: paddle.Tensor = kernel
    return kernel_2d


def laplacian_hq_loss(prediction, gt):
    b, c, h, w = prediction.shape
    kernel_size = 3
    kernel = (
        get_laplacian_kernel2d(kernel_size).to(prediction.place).to(prediction.dtype)
    )
    kernel = kernel.tile(repeat_times=[c, 1, 1, 1])
    padding = (kernel_size - 1) // 2
    lap_pre = paddle.nn.functional.conv2d(
        x=prediction, weight=kernel, padding=padding, stride=1, groups=c
    )
    lap_gt = paddle.nn.functional.conv2d(
        x=gt, weight=kernel, padding=padding, stride=1, groups=c
    )
    return ((lap_pre - lap_gt) ** 2).mean()


class VGG19FeatLayer(paddle.nn.Layer):
    def __init__(self):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval()
        self.mean = paddle.to_tensor(data=[0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
        self.std = paddle.to_tensor(data=[0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])

    def forward(self, x):
        out = {}
        x = x - self.mean
        x = x / self.std
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, paddle.nn.Conv2D):
                ri += 1
                name = "conv{}_{}".format(ci, ri)
            elif isinstance(layer, paddle.nn.ReLU):
                ri += 1
                name = "relu{}_{}".format(ci, ri)
                layer = paddle.nn.ReLU()
            elif isinstance(layer, paddle.nn.MaxPool2D):
                ri = 0
                name = "pool_{}".format(ci)
                ci += 1
            elif isinstance(layer, paddle.nn.BatchNorm2D):
                name = "bn_{}".format(ci)
            else:
                raise RuntimeError(
                    "Unrecognized layer: {}".format(layer.__class__.__name__)
                )
            x = layer(x)
            out[name] = x
        return out


class IDMRFLoss(paddle.nn.Layer):
    def __init__(self, featlayer=VGG19FeatLayer):
        super(IDMRFLoss, self).__init__()
        self.featlayer = featlayer()
        self.feat_style_layers = {"relu3_2": 1.0, "relu4_2": 1.0}
        self.feat_content_layers = {"relu4_2": 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = paddle.sum(x=featmaps, axis=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = paddle_add.paddle_unfold(
            paddle_add.paddle_unfold(featmaps, 2, patch_size, patch_stride),
            3,
            patch_size,
            patch_stride,
        )
        self.patches_OIHW = patches_as_depth_vectors.transpose(perm=[0, 2, 3, 1, 4, 5])
        dims = self.patches_OIHW.shape
        self.patches_OIHW = self.patches_OIHW.reshape([-1, dims[3], dims[4], dims[5]])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-05
        div = (
            paddle.min(keepdim=True, x=cdist, axis=1),
            paddle.argmin(keepdim=True, x=cdist, axis=1),
        )[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = paddle.exp(
            x=(self.bias - scaled_dist) / self.nn_stretch_sigma
        )
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = paddle.mean(x=tar, axis=1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT
        gen_feats_norm = paddle.linalg.norm(x=gen_feats, p=2, axis=1, keepdim=True)
        tar_feats_norm = paddle.linalg.norm(x=tar_feats, p=2, axis=1, keepdim=True)
        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm
        cosine_dist_l = []
        BatchSize = tar.shape[0]
        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i : i + 1, :, :, :]
            gen_feat_i = gen_normalized[i : i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)
            cosine_dist_i = paddle.nn.functional.conv2d(
                x=gen_feat_i, weight=patches_OIHW
            )
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = paddle.concat(x=cosine_dist_l, axis=0)
        cosine_dist_zero_2_one = -(cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.shape
        k_max_nc = (
            paddle.max(
                x=rela_dist.reshape([dims_div_mrf[0], dims_div_mrf[1], -1]), axis=2
            ),
            paddle.argmax(
                x=rela_dist.reshape([dims_div_mrf[0], dims_div_mrf[1], -1]), axis=2
            ),
        )[0]
        div_mrf = paddle.mean(x=k_max_nc, axis=1)
        div_mrf_sum = -paddle.log(x=div_mrf)
        div_mrf_sum = paddle.sum(x=div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [
            (
                self.feat_style_layers[layer]
                * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])
            )
            for layer in self.feat_style_layers
        ]
        self.style_loss = (
            reduce(lambda x, y: x + y, style_loss_list) * self.lambda_style
        )
        content_loss_list = [
            (
                self.feat_content_layers[layer]
                * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])
            )
            for layer in self.feat_content_layers
        ]
        self.content_loss = (
            reduce(lambda x, y: x + y, content_loss_list) * self.lambda_content
        )
        return self.style_loss + self.content_loss


class VGG_16(paddle.nn.Layer):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = paddle.nn.Conv2D(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv_1_2 = paddle.nn.Conv2D(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv_2_1 = paddle.nn.Conv2D(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv_2_2 = paddle.nn.Conv2D(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv_3_1 = paddle.nn.Conv2D(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv_3_2 = paddle.nn.Conv2D(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv_3_3 = paddle.nn.Conv2D(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv_4_1 = paddle.nn.Conv2D(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv_4_2 = paddle.nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv_4_3 = paddle.nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv_5_1 = paddle.nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv_5_2 = paddle.nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv_5_3 = paddle.nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.fc6 = paddle.nn.Linear(in_features=512 * 7 * 7, out_features=4096)
        self.fc7 = paddle.nn.Linear(in_features=4096, out_features=4096)
        self.fc8 = paddle.nn.Linear(in_features=4096, out_features=2622)
        self.mean = (
            paddle.to_tensor(
                data=np.array([129.1863, 104.7624, 93.594]) / 255.0, dtype="float32"
            )
            .astype(dtype="float32")
            .reshape([1, 3, 1, 1])
        )

    def load_weights(self, path="pretrained/VGG_FACE.t7"):
        """Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        print(model)
        exit()
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = paddle.to_tensor(
                        data=layer.weight
                    ).reshape(self_layer.weight.shape)[...]
                    self_layer.bias.data[...] = paddle.to_tensor(
                        data=layer.bias
                    ).reshape(self_layer.bias.shape)[...]
                else:
                    self_layer = getattr(self, "fc%d" % block)
                    block += 1
                    self_layer.weight.data[...] = paddle.to_tensor(
                        data=layer.weight
                    ).reshape(self_layer.weight.shape)[...]
                    self_layer.bias.data[...] = paddle.to_tensor(
                        data=layer.bias
                    ).reshape(self_layer.bias.shape)[...]

    def forward(self, x):
        """Pytorch forward
        Args:
            x: input image (224x224)
        Returns: class logits
        """
        out = {}
        x = x - self.mean
        x = paddle.nn.functional.relu(x=self.conv_1_1(x))
        x = paddle.nn.functional.relu(x=self.conv_1_2(x))
        x = paddle.nn.functional.max_pool2d(kernel_size=2, stride=2, x=x)
        x = paddle.nn.functional.relu(x=self.conv_2_1(x))
        x = paddle.nn.functional.relu(x=self.conv_2_2(x))
        x = paddle.nn.functional.max_pool2d(kernel_size=2, stride=2, x=x)
        x = paddle.nn.functional.relu(x=self.conv_3_1(x))
        x = paddle.nn.functional.relu(x=self.conv_3_2(x))
        out["relu3_2"] = x
        x = paddle.nn.functional.relu(x=self.conv_3_3(x))
        x = paddle.nn.functional.max_pool2d(kernel_size=2, stride=2, x=x)
        x = paddle.nn.functional.relu(x=self.conv_4_1(x))
        x = paddle.nn.functional.relu(x=self.conv_4_2(x))
        out["relu4_2"] = x
        x = paddle.nn.functional.relu(x=self.conv_4_3(x))
        x = paddle.nn.functional.max_pool2d(kernel_size=2, stride=2, x=x)
        x = paddle.nn.functional.relu(x=self.conv_5_1(x))
        x = paddle.nn.functional.relu(x=self.conv_5_2(x))
        x = paddle.nn.functional.relu(x=self.conv_5_3(x))
        x = paddle.nn.functional.max_pool2d(kernel_size=2, stride=2, x=x)
        x = x.reshape([x.shape[0], -1])
        x = paddle.nn.functional.relu(x=self.fc6(x))
        x = paddle.nn.functional.dropout(x=x, p=0.5, training=self.training)
        x = paddle.nn.functional.relu(x=self.fc7(x))
        x = paddle.nn.functional.dropout(x=x, p=0.5, training=self.training)
        x = self.fc8(x)
        out["last"] = x
        return out


class VGGLoss(paddle.nn.Layer):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.featlayer = VGG_16().astype(dtype="float32")
        self.featlayer.load_weights(
            path="data/face_recognition_model/vgg_face_torch/VGG_FACE.t7"
        )
        self.featlayer = self.featlayer.eval()
        self.feat_style_layers = {"relu3_2": 1.0, "relu4_2": 1.0}
        self.feat_content_layers = {"relu4_2": 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = paddle.sum(x=featmaps, axis=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = paddle_add.paddle_unfold(
            paddle_add.paddle_unfold(featmaps, 2, patch_size, patch_stride),
            3,
            patch_size,
            patch_stride,
        )
        self.patches_OIHW = patches_as_depth_vectors.transpose(perm=[0, 2, 3, 1, 4, 5])
        dims = self.patches_OIHW.shape
        self.patches_OIHW = self.patches_OIHW.reshape([-1, dims[3], dims[4], dims[5]])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-05
        div = (
            paddle.min(keepdim=True, x=cdist, axis=1),
            paddle.argmin(keepdim=True, x=cdist, axis=1),
        )[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = paddle.exp(
            x=(self.bias - scaled_dist) / self.nn_stretch_sigma
        )
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = paddle.mean(x=tar, axis=1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT
        gen_feats_norm = paddle.linalg.norm(x=gen_feats, p=2, axis=1, keepdim=True)
        tar_feats_norm = paddle.linalg.norm(x=tar_feats, p=2, axis=1, keepdim=True)
        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm
        cosine_dist_l = []
        BatchSize = tar.shape[0]
        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i : i + 1, :, :, :]
            gen_feat_i = gen_normalized[i : i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)
            cosine_dist_i = paddle.nn.functional.conv2d(
                x=gen_feat_i, weight=patches_OIHW
            )
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = paddle.concat(x=cosine_dist_l, axis=0)
        cosine_dist_zero_2_one = -(cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.shape
        k_max_nc = (
            paddle.max(
                x=rela_dist.reshape([dims_div_mrf[0], dims_div_mrf[1], -1]), axis=2
            ),
            paddle.argmax(
                x=rela_dist.reshape([dims_div_mrf[0], dims_div_mrf[1], -1]), axis=2
            ),
        )[0]
        div_mrf = paddle.mean(x=k_max_nc, axis=1)
        div_mrf_sum = -paddle.log(x=div_mrf)
        div_mrf_sum = paddle.sum(x=div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [
            (
                self.feat_style_layers[layer]
                * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])
            )
            for layer in self.feat_style_layers
        ]
        self.style_loss = (
            reduce(lambda x, y: x + y, style_loss_list) * self.lambda_style
        )
        content_loss_list = [
            (
                self.feat_content_layers[layer]
                * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])
            )
            for layer in self.feat_content_layers
        ]
        self.content_loss = (
            reduce(lambda x, y: x + y, content_loss_list) * self.lambda_content
        )
        return self.style_loss + self.content_loss


# ref: https://github.com/cydonia999/VGGFace2-pytorch
from ..models.frnet import load_state_dict, resnet50


class VGGFace2Loss(paddle.nn.Layer):
    def __init__(self, pretrained_model, pretrained_data="vggface2"):
        super(VGGFace2Loss, self).__init__()
        self.reg_model = resnet50(num_classes=8631, include_top=False).eval().cuda()
        load_state_dict(self.reg_model, pretrained_model)
        self.mean_bgr = paddle.to_tensor(data=[91.4953, 103.8827, 131.0912])

    def reg_features(self, x):
        margin = 10
        x = x[:, :, margin : 224 - margin, margin : 224 - margin]
        x = paddle.nn.functional.interpolate(
            x=x * 2.0 - 1.0, size=[224, 224], mode="bilinear"
        )
        feature = self.reg_model(x)
        feature = feature.reshape([x.shape[0], -1])
        return feature

    def transform(self, img):
        img = (
            img[:, ([2, 1, 0]), :, :].transpose(perm=[0, 2, 3, 1]) * 255 - self.mean_bgr
        )
        img = img.transpose(perm=[0, 3, 1, 2])
        return img

    def _cos_metric(self, x1, x2):
        return 1.0 - paddle.nn.functional.cosine_similarity(x1=x1, x2=x2, axis=1)

    def forward(self, gen, tar, is_crop=True):
        gen = self.transform(gen)
        tar = self.transform(tar)
        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)
        loss = self._cos_metric(gen_out, tar_out).mean()
        return loss
