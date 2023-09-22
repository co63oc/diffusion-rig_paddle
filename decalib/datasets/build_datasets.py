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

import paddle

from .aflw2000 import AFLW2000
from .ethnicity import EthnicityDataset
from .now import NoWDataset
from .vggface import VGGFace2Dataset
from .vox import VoxelDataset


def build_train(config, is_train=True):
    data_list = []
    if "vox2" in config.training_data:
        data_list.append(
            VoxelDataset(
                dataname="vox2",
                K=config.K,
                image_size=config.image_size,
                scale=[config.scale_min, config.scale_max],
                trans_scale=config.trans_scale,
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
    dataset = paddle.io.ComposeDataset(data_list)
    return dataset


def build_val(config, is_train=True):
    data_list = []
    if "vggface2" in config.eval_data:
        data_list.append(
            VGGFace2Dataset(
                isEval=True,
                K=config.K,
                image_size=config.image_size,
                scale=[config.scale_min, config.scale_max],
                trans_scale=config.trans_scale,
                isSingle=config.isSingle,
            )
        )
    if "now" in config.eval_data:
        data_list.append(NoWDataset())
    if "aflw2000" in config.eval_data:
        data_list.append(AFLW2000())
    dataset = paddle.io.ComposeDataset(data_list)
    return dataset
