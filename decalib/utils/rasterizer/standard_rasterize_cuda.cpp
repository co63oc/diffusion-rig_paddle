// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/extension.h"
#include <vector>
#include <iostream>

std::vector<at::Tensor> forward_rasterize_cuda(
        at::Tensor face_vertices,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor baryw_buffer,
        int h,
        int w);

std::vector<at::Tensor> standard_rasterize(
        at::Tensor face_vertices,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor baryw_buffer,
        int height, int width
        ) {
    return forward_rasterize_cuda(face_vertices, depth_buffer, triangle_buffer, baryw_buffer, height, width);
}

std::vector<at::Tensor> forward_rasterize_colors_cuda(
        at::Tensor face_vertices,
        at::Tensor face_colors,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor images,
        int h,
        int w);

std::vector<at::Tensor> standard_rasterize_colors(
        at::Tensor face_vertices,
        at::Tensor face_colors,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor images,
        int height, int width
        ) {
    return forward_rasterize_colors_cuda(face_vertices, face_colors, depth_buffer, triangle_buffer, images, height, width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("standard_rasterize", &standard_rasterize, "RASTERIZE (CUDA)");
    m.def("standard_rasterize_colors", &standard_rasterize_colors, "RASTERIZE COLORS (CUDA)");
}

// TODO: backward