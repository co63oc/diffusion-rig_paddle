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
import argparse
import os
import sys

from diffusion.resample import create_named_schedule_sampler
from utils import dist_util, logger
from utils.image_datasets import load_data, load_data_local
from utils.script_util import (add_dict_to_argparser, args_to_dict,
                               create_model_and_diffusion,
                               model_and_diffusion_defaults)
from utils.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)
    logger.log(args)
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    logger.log("creating data loader...")
    if args.stage == 1:
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.stage == 2:
        data = load_data_local(data_dir=args.data_dir, batch_size=args.batch_size)
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        stage=args.stage,
        max_steps=args.max_steps,
        auto_scale_grad_clip=args.auto_scale_grad_clip,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=0.0001,
        weight_decay=0.0,
        batch_size=1,
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        log_dir="stage1",
        num_workers=16,
        max_steps=0,
        auto_scale_grad_clip=1.0,
        stage=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
