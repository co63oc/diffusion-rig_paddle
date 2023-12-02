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

std::vector<paddle::Tensor> forward_rasterize_cuda(
        paddle::Tensor face_vertices,
        paddle::Tensor depth_buffer,
        paddle::Tensor triangle_buffer,
        paddle::Tensor baryw_buffer,
        int h,
        int w);

std::vector<paddle::Tensor> standard_rasterize(
        paddle::Tensor face_vertices,
        paddle::Tensor depth_buffer,
        paddle::Tensor triangle_buffer,
        paddle::Tensor baryw_buffer,
        int height, int width
        ) {
    return forward_rasterize_cuda(face_vertices, depth_buffer, triangle_buffer, baryw_buffer, height, width);
}

std::vector<paddle::Tensor> forward_rasterize_colors_cuda(
        paddle::Tensor face_vertices,
        paddle::Tensor face_colors,
        paddle::Tensor depth_buffer,
        paddle::Tensor triangle_buffer,
        paddle::Tensor images,
        int h,
        int w);

std::vector<paddle::Tensor> standard_rasterize_colors(
        paddle::Tensor face_vertices,
        paddle::Tensor face_colors,
        paddle::Tensor depth_buffer,
        paddle::Tensor triangle_buffer,
        paddle::Tensor images,
        int height, int width
        ) {
    return forward_rasterize_colors_cuda(face_vertices, face_colors, depth_buffer, triangle_buffer, images, height, width);
}

PYBIND11_MODULE(standard_rasterize_cuda, m) {
    m.def("standard_rasterize", &standard_rasterize, "RASTERIZE (CUDA)");
    m.def("standard_rasterize_colors", &standard_rasterize_colors, "RASTERIZE COLORS (CUDA)");
}

// TODO: backward