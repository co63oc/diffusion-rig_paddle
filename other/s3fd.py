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
import torch

model = torch.load("s3fd-619a316812.pth")

model_path = "s3fd-619a316812.pth"
paddle_model_path = "paddle_s3fd-619a316812.pth"
model_path = "2DFAN4-11f355bf06.pth.tar"
paddle_model_path = "paddle_2DFAN4-11f355bf06.pth.tar"
model_path = "3DFAN4-7835d9f11d.pth.tar"
paddle_model_path = "paddle_3DFAN4-7835d9f11d.pth.tar"
checkpoint = torch.load(model_path)

paddle_checkpoint = {}
for k, v in checkpoint.items():
    paddle_checkpoint[k] = paddle.to_tensor(v.cpu().numpy())

paddle.save(paddle_checkpoint, paddle_model_path)
print("Convert finish.")
