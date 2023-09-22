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
import pickle
import sys
from datetime import datetime
from time import time

import cv2
import numpy as np
import paddle
from loguru import logger
from skimage.io import imread
from tqdm import tqdm

from .datasets import datasets
from .models.decoders import Generator
from .models.encoders import ResnetEncoder
from .models.FLAME import FLAME, FLAMETex
from .utils import util
from .utils.config import cfg
from .utils.renderer import SRenderY
from .utils.rotation_converter import batch_euler2axis

torch.backends.cudnn.benchmark = True
from .datasets import build_datasets
from .utils import lossfunc


class Trainer(object):
    def __init__(self, model, config=None, device="cuda:0"):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.place = device
        self.batch_size = self.cfg.dataset.batch_size
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self.deca = model
        self.E_flame = self.deca.E_flame
        self.flametex = self.deca.flametex
        self.configure_optimizers()
        self.load_checkpoint()
        self.mrf_loss = lossfunc.IDMRFLoss()
        self.id_loss = lossfunc.VGGFace2Loss()
        self.face_attr_mask = util.load_local_mask(
            image_size=self.cfg.model.uv_size, mode="bbx"
        )
        logger.add(
            os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, "train.log")
        )
        # if self.cfg.train.write_summary:
        #     TODO: not convert
        #     self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=os.
        #         path.join(self.cfg.output_dir, self.cfg.train.log_dir))

    def configure_optimizers(self):
        self.opt = paddle.optimizer.Adam(
            parameters=self.E_flame.parameters(),
            learning_rate=self.cfg.train.lr,
            amsgrad=False,
        )

    def load_checkpoint(self):
        model_dict = self.deca.model_dict()
        if self.cfg.train.resume and os.path.exists(
            os.path.join(self.cfg.output_dir, "model.tar")
        ):
            checkpoint = paddle.load(
                path=os.path.join(self.cfg.output_dir, "model.tar")
            )
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    util.copy_state_dict(model_dict[key], checkpoint[key])
            util.copy_state_dict(self.opt.state_dict(), checkpoint["opt"])
            self.global_step = checkpoint["global_step"]
            logger.info(
                f"resume training from {os.path.join(self.cfg.output_dir, 'model.tar')}"
            )
            logger.info(f"training start from step {self.global_step}")
        elif os.path.exists(self.cfg.pretrained_modelpath):
            checkpoint = paddle.load(path=self.cfg.pretrained_modelpath)
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    util.copy_state_dict(model_dict[key], checkpoint[key])
            self.global_step = 0
        else:
            logger.info("model path not found, start training from scratch")
            self.global_step = 0

    def training_step(self, batch, batch_nb):
        self.deca.train()
        images = batch["image"]
        images = images.reshape(
            [-1, images.shape[-3], images.shape[-2], images.shape[-1]]
        )
        lmk = batch["landmark"]
        lmk = lmk.reshape([-1, lmk.shape[-2], lmk.shape[-1]])
        masks = batch["mask"]
        masks = masks.reshape([-1, images.shape[-2], images.shape[-1]])
        codedict = self.deca.encode(images)
        if self.cfg.loss.shape_consistency == "exchange":
            """
            make sure s0, s1 is something to make shape close
            the difference from ||so - s1|| is
            the later encourage s0, s1 is cloase in l2 space, but not really ensure shape will be close
            """
            new_order = np.array(
                [
                    (np.random.permutation(self.K) + i * self.K)
                    for i in range(self.batch_size)
                ]
            )
            new_order = new_order.flatten()
            shapecode = codedict["shape"]
            shapecode_new = shapecode[new_order]
            codedict[key] = paddle.concat(x=[shapecode, shapecode_new], axis=0)
            for key in ["tex", "exp", "pose", "cam", "light", "images"]:
                code = codedict[key]
                codedict[key] = paddle.concat(x=[code, code], axis=0)
            images = paddle.concat(x=[images, images], axis=0)
            lmk = paddle.concat(x=[lmk, lmk], axis=0)
            masks = paddle.concat(x=[masks, masks], axis=0)
        batch_size = images.shape[0]
        opdict = self.deca.decode(
            codedict, vis_lmk=False, return_vis=False, use_detail=False
        )
        mask_face_eye = paddle.nn.functional.grid_sample(
            x=self.deca.uv_face_eye_mask.expand(shape=[batch_size, -1, -1, -1]),
            grid=opdict["grid"].detach(),
            align_corners=False,
        )
        predicted_images = (
            opdict["rendered_images"] * mask_face_eye * opdict["alpha_images"]
        )
        opdict["predicted_images"] = predicted_images
        opdict["images"] = images
        opdict["lmk"] = lmk
        losses = {}
        predicted_landmarks = opdict["landmarks2d"]
        if self.cfg.loss.useWlmk:
            losses["landmark"] = (
                lossfunc.weighted_landmark_loss(predicted_landmarks, lmk)
                * self.cfg.loss.lmk
            )
        else:
            losses["landmark"] = (
                lossfunc.landmark_loss(predicted_landmarks, lmk) * self.cfg.loss.lmk
            )
        if self.cfg.loss.eyed > 0.0:
            losses["eye_distance"] = (
                lossfunc.eyed_loss(predicted_landmarks, lmk) * self.cfg.loss.eyed
            )
        if self.cfg.loss.lipd > 0.0:
            losses["lip_distance"] = (
                lossfunc.lipd_loss(predicted_landmarks, lmk) * self.cfg.loss.lipd
            )
        if self.cfg.loss.useSeg:
            masks = masks[:, (None), :, :]
        else:
            masks = mask_face_eye * opdict["alpha_images"]
        if self.cfg.loss.photo > 0.0:
            losses["photometric_texture"] = (
                masks * (predicted_images - images).abs()
            ).mean() * self.cfg.loss.photo
        if self.cfg.loss.id > 0.0:
            shading_images = self.deca.render.add_SHlight(
                opdict["normal_images"], codedict["light"].detach()
            )
            albedo_images = paddle.nn.functional.grid_sample(
                x=opdict["albedo"].detach(), grid=opdict["grid"], align_corners=False
            )
            overlay = albedo_images * shading_images * mask_face_eye + images * (
                1 - mask_face_eye
            )
            losses["identity"] = self.id_loss(overlay, images) * self.cfg.loss.id
        losses["shape_reg"] = (
            paddle.sum(x=codedict["shape"] ** 2) / 2 * self.cfg.loss.reg_shape
        )
        losses["expression_reg"] = (
            paddle.sum(x=codedict["exp"] ** 2) / 2 * self.cfg.loss.reg_exp
        )
        losses["tex_reg"] = (
            paddle.sum(x=codedict["tex"] ** 2) / 2 * self.cfg.loss.reg_tex
        )
        losses["light_reg"] = (
            (paddle.mean(x=codedict["light"], axis=2)[:, :, (None)] - codedict["light"])
            ** 2
        ).mean() * self.cfg.loss.reg_light
        if self.cfg.model.jaw_type == "euler":
            losses["reg_jawpose_roll"] = (
                paddle.sum(x=codedict["euler_jaw_pose"][:, (-1)] ** 2) / 2 * 10.0
            )
            losses["reg_jawpose_close"] = (
                paddle.sum(
                    x=paddle.nn.functional.relu(x=-codedict["euler_jaw_pose"][:, (0)])
                    ** 2
                )
                / 2
                * 10.0
            )
        all_loss = 0.0
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses["all_loss"] = all_loss
        return losses, opdict

    def validation_step(self):
        self.deca.eval()
        try:
            batch = next(self.val_iter)
        except:
            self.val_iter = iter(self.val_dataloader)
            batch = next(self.val_iter)
        images = batch["image"]
        images = images.reshape(
            [-1, images.shape[-3], images.shape[-2], images.shape[-1]]
        )
        with paddle.no_grad():
            codedict = self.deca.encode(images)
            opdict, visdict = self.deca.decode(codedict)
        savepath = os.path.join(
            self.cfg.output_dir,
            self.cfg.train.val_vis_dir,
            f"{self.global_step:08}.jpg",
        )
        util.visualize_grid(visdict, savepath)

    def evaluate(self):
        """NOW validation"""
        os.makedirs(os.path.join(self.cfg.output_dir, "NOW_validation"), exist_ok=True)
        savefolder = os.path.join(
            self.cfg.output_dir, "NOW_validation", f"step_{self.global_step:08}"
        )
        os.makedirs(savefolder, exist_ok=True)
        self.deca.eval()
        from .datasets.now import NoWDataset

        dataset = NoWDataset(
            scale=(self.cfg.dataset.scale_min + self.cfg.dataset.scale_max) / 2
        )
        dataloader = paddle.io.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=8, drop_last=False
        )
        faces = self.deca.flame.faces_tensor.cpu().numpy()
        for i, batch in enumerate(tqdm(dataloader, desc="now evaluation ")):
            images = batch["image"]
            imagename = batch["imagename"]
            with paddle.no_grad():
                codedict = self.deca.encode(images)
                codedict["exp"][:] = 0.0
                codedict["pose"][:] = 0.0
                opdict, visdict = self.deca.decode(codedict)
            verts = opdict["verts"].cpu().numpy()
            landmark_51 = opdict["landmarks3d_world"][:, 17:]
            landmark_7 = landmark_51[:, ([19, 22, 25, 28, 16, 31, 37])]
            landmark_7 = landmark_7.cpu().numpy()
            for k in range(images.shape[0]):
                os.makedirs(os.path.join(savefolder, imagename[k]), exist_ok=True)
                util.write_obj(
                    os.path.join(savefolder, f"{imagename[k]}.obj"),
                    vertices=verts[k],
                    faces=faces,
                )
                np.save(os.path.join(savefolder, f"{imagename[k]}.npy"), landmark_7[k])
            util.visualize_grid(visdict, os.path.join(savefolder, f"{i}.jpg"))

    def prepare_data(self):
        self.train_dataset = build_datasets.build_train(self.cfg.dataset)
        self.val_dataset = build_datasets.build_val(self.cfg.dataset)
        logger.info("---- training data numbers: ", len(self.train_dataset))
        self.train_dataloader = paddle.io.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
        )
        self.val_dataloader = paddle.io.DataLoader(
            self.val_dataset, batch_size=8, shuffle=True, num_workers=8, drop_last=False
        )
        self.val_iter = iter(self.val_dataloader)

    def fit(self):
        self.prepare_data()
        iters_every_epoch = int(len(self.train_dataset) / self.batch_size)
        start_epoch = self.global_step // iters_every_epoch
        for epoch in tqdm(range(start_epoch, self.cfg.train.max_epochs)):
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                losses, opdict = self.training_step(batch, step)
                import ipdb

                ipdb.set_trace()
                if self.global_step % self.cfg.train.log_steps == 0:
                    loss_info = f"""ExpName: {self.cfg.exp_name} 
Epoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} 
"""
                    for k, v in losses.items():
                        loss_info = loss_info + f"{k}: {v:.4f}, "
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar(
                                "train_loss/" + k, v, global_step=self.global_step
                            )
                    logger.info(loss_info)
                if self.global_step % self.cfg.train.vis_steps == 0:
                    visind = list(range(8))
                    shape_images = self.deca.render.render_shape(
                        opdict["verts"], opdict["trans_verts"]
                    )
                    visdict = {
                        "inputs": opdict["images"][visind],
                        "landmarks2d_gt": util.tensor_vis_landmarks(
                            opdict["images"][visind],
                            opdict["lmk"][visind],
                            isScale=True,
                        ),
                        "landmarks2d": util.tensor_vis_landmarks(
                            opdict["images"][visind],
                            opdict["landmarks2d"][visind],
                            isScale=True,
                        ),
                        "shape_images": shape_images[visind],
                        "predicted_images": opdict["predicted_images"][visind],
                    }
                    savepath = os.path.join(
                        self.cfg.output_dir,
                        self.cfg.train.vis_dir,
                        f"{self.global_step:06}.jpg",
                    )
                    util.visualize_grid(visdict, savepath)
                if self.global_step % self.cfg.train.checkpoint_steps == 0:
                    model_dict = self.deca.model_dict()
                    model_dict["opt"] = self.opt.state_dict()
                    model_dict["global_step"] = self.global_step
                    model_dict["batch_size"] = self.batch_size
                    paddle.save(
                        obj=model_dict,
                        path=os.path.join(self.cfg.output_dir, "model" + ".tar"),
                        protocol=4,
                    )
                if self.global_step % self.cfg.train.val_steps == 0:
                    self.validation_step()
                if self.global_step % self.cfg.train.eval_steps == 0:
                    self.evaluate()
                all_loss = losses["all_loss"]
                self.opt.clear_grad()
                all_loss.backward()
                self.opt.step()
                self.global_step += 1
                if self.global_step > self.cfg.train.max_steps:
                    break