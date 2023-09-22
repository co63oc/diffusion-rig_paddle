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
import pickle
from io import BytesIO

import lmdb
import paddle
from PIL import Image
from tqdm import tqdm

from decalib.datasets import datasets
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from utils.script_util import add_dict_to_argparser


def main():
    args = create_argparser().parse_args()
    deca_cfg.model.use_tex = True
    deca_cfg.model.tex_path = "data/FLAME_texture.npz"
    deca_cfg.model.tex_type = "FLAME"
    deca = DECA(config=deca_cfg, device=paddle.CUDAPlace(0))
    dataset_root = args.data_dir
    testdata = datasets.TestData(
        dataset_root, iscrop=True, size=args.image_size, sort=True
    )
    batch_size = args.batch_size
    loader = paddle.io.DataLoader(testdata, batch_size=batch_size)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if args.use_meanshape:
        shapes = []
        for td in testdata:
            img = td["image"].cuda().unsqueeze(axis=0)
            code = deca.encode(img)
            shapes.append(code["shape"].detach())
        mean_shape = paddle.mean(
            x=paddle.concat(x=shapes, axis=0), axis=0, keepdim=True
        )
        with open(os.path.join(output_dir, "mean_shape.pkl"), "wb") as f:
            pickle.dump(mean_shape, f)
    with lmdb.open(output_dir, map_size=1024**4, readahead=False) as env:
        total = 0
        for batch_id, data in enumerate(tqdm(loader)):
            with paddle.no_grad():
                inp = data["image"].cuda()
                codedict = deca.encode(inp)
                tform = data["tform"]
                x = paddle.linalg.inv(x=tform)
                perm_2 = list(range(x.ndim))
                perm_2[1] = 2
                perm_2[2] = 1
                tform = x.transpose(perm=perm_2).cuda()
                original_image = data["original_image"].cuda()
                if args.use_meanshape:
                    codedict["shape"] = mean_shape.tile(repeat_times=[inp.shape[0], 1])
                codedict["tform"] = tform
                opdict, _ = deca.decode(
                    codedict,
                    render_orig=True,
                    original_image=original_image,
                    tform=tform,
                )
                opdict["inputs"] = original_image
                for item_id in range(inp.shape[0]):
                    i = batch_id * batch_size + item_id
                    image = (
                        (original_image[item_id].detach().cpu().numpy() * 255)
                        .astype("uint8")
                        .transpose((1, 2, 0))
                    )
                    image = Image.fromarray(image)
                    albedo_key = f"albedo_{str(i).zfill(6)}".encode("utf-8")
                    buffer = BytesIO()
                    pickle.dump(opdict["albedo_images"][item_id].detach().cpu(), buffer)
                    albedo_val = buffer.getvalue()
                    normal_key = f"normal_{str(i).zfill(6)}".encode("utf-8")
                    buffer = BytesIO()
                    pickle.dump(opdict["normal_images"][item_id].detach().cpu(), buffer)
                    normal_val = buffer.getvalue()
                    rendered_key = f"rendered_{str(i).zfill(6)}".encode("utf-8")
                    buffer = BytesIO()
                    pickle.dump(
                        opdict["rendered_images"][item_id].detach().cpu(), buffer
                    )
                    rendered_val = buffer.getvalue()
                    image_key = f"image_{str(i).zfill(6)}".encode("utf-8")
                    buffer = BytesIO()
                    image.save(buffer, format="png", quality=100)
                    image_val = buffer.getvalue()
                    with env.begin(write=True) as transaction:
                        transaction.put(albedo_key, albedo_val)
                        transaction.put(normal_key, normal_val)
                        transaction.put(rendered_key, rendered_val)
                        transaction.put(image_key, image_val)
                    total += 1
        with env.begin(write=True) as transaction:
            transaction.put("length".encode("utf-8"), str(total).encode("utf-8"))


def create_argparser():
    defaults = dict(
        data_dir="", output_dir="", image_size=256, batch_size=8, use_meanshape=False
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
