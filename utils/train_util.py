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

import datetime
import functools
import os
import time

import blobfile as bf
import paddle

from diffusion.resample import LossAwareSampler, UniformSampler
from utils import dist_util

from . import logger


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        log_interval,
        save_interval,
        resume_checkpoint,
        schedule_sampler=None,
        weight_decay=0.0,
        stage=1,
        max_steps=0,
        auto_scale_grad_clip=1.0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.stage = stage
        self.max_steps = max_steps
        self.auto_scale_grad_clip = auto_scale_grad_clip
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * paddle.distributed.get_world_size()
        self.sync_cuda = paddle.device.cuda.device_count() >= 1
        self._load_and_sync_parameters()
        if stage == 2:
            for n, p in self.model.named_parameters():
                if not n.startswith("denoisingUNet."):
                    p.stop_gradient = not False
        self.opt = paddle.optimizer.Adam(
            parameters=self.model.parameters(),
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scaler = paddle.amp.GradScaler(
            incr_every_n_steps=2000, init_loss_scaling=65536.0
        )
        if self.resume_step:
            self._load_optimizer_state()
        if paddle.device.cuda.device_count() >= 1:
            self.use_ddp = True
            # self.ddp_model = (torch.nn.parallel.distributed.
            #     DistributedDataParallel(self.model, device_ids=[dist_util.
            #     dev()], output_device=dist_util.dev(), broadcast_buffers=
            #     False, bucket_cap_mb=128, find_unused_parameters=False))
            self.ddp_model = paddle.DataParallel(
                self.model, comm_buffer_size=128, find_unused_parameters=False
            )
        else:
            if paddle.distributed.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        self.start_time = time.time()
        self.step_time = 0
        self.last_step_time = None

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            if self.stage == 1:
                self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            ckpt = dist_util.load_state_dict(
                resume_checkpoint, map_location=dist_util.dev()
            )
            self.model.set_state_dict(state_dict=ckpt)
        dist_util.sync_params(self.model.parameters())

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.set_state_dict(state_dict=state_dict)

    def run_loop(self):
        while not self.max_steps or self.step + self.resume_step < self.max_steps:
            batch = next(self.data)
            image = batch["image"]
            rendered, normal, albedo = (
                batch["rendered"],
                batch["normal"],
                batch["albedo"],
            )
            physic_cond = paddle.concat(x=[rendered, normal, albedo], axis=1)
            self.run_step(image, physic_cond)
            if (self.step + self.resume_step) % self.log_interval == 0:
                logger.log(logger.get_dir())
                logger.dumpkvs()
            if (
                self.step + self.resume_step
            ) % self.save_interval == 0 and self.step > 0:
                self.save()
                logger.log(logger.get_dir())
                logger.dumpkvs()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        if (self.step + self.resume_step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, image, physic_cond):
        self.forward_backward(image, physic_cond)
        self.scaler.unscale_(self.opt)
        paddle.nn.utils.clip_grad_norm_(
            parameters=self.ddp_model.parameters(), max_norm=self.auto_scale_grad_clip
        )
        self.scaler.step(self.opt)
        self.scaler.update()
        self.log_step()

    def forward_backward(self, image, physic_cond):
        self.opt.clear_grad()
        image = image.to(dist_util.dev())
        physic_cond = physic_cond.to(dist_util.dev())
        t, weights = self.schedule_sampler.sample(image.shape[0], dist_util.dev())
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            image,
            t,
            model_kwargs={"physic_cond": physic_cond, "x_start": image},
        )
        with paddle.amp.auto_cast():
            losses = compute_losses()
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            loss = (losses["loss"] * weights).mean()
        log_loss_dict(self.diffusion, t, {k: (v * weights) for k, v in losses.items()})
        self.scaler.scale(loss).backward()

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        logger.logkv(
            "time elapsed",
            str(datetime.timedelta(seconds=time.time() - self.start_time)),
        )
        ct = time.time()
        if self.step_time == 0:
            self.step_time = ct - self.start_time
            self.last_step_time = ct
        else:
            self.step_time = 0.1 * self.step_time + 0.9 * (ct - self.last_step_time)
            self.last_step_time = ct
        logger.logkv(
            "time est.(10k)", str(datetime.timedelta(seconds=10000 * self.step_time))
        )
        steps_to_go = (
            self.save_interval - self.step - self.resume_step
        ) % self.save_interval
        logger.logkv(
            "time est.(next ckpt)",
            str(datetime.timedelta(seconds=steps_to_go * self.step_time)),
        )

    def save(self):
        def save_checkpoint(rate, params=None):
            state_dict = self.model.state_dict()
            if paddle.distributed.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{self.step + self.resume_step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    paddle.save(obj=state_dict, path=f, protocol=4)

        save_checkpoint(0)
        if paddle.distributed.get_rank() == 0 and self.stage == 1:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{self.step + self.resume_step:06d}.pt"),
                "wb",
            ) as f:
                paddle.save(obj=self.opt.state_dict(), path=f, protocol=4)
        paddle.distributed.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
