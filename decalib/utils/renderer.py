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

import numpy as np
import paddle

from . import util


def set_rasterizer(type="paddle3d"):
    if type == "paddle3d":
        global Meshes, load_obj, rasterize_meshes
        from paddle3d.io import load_obj
        from paddle3d.renderer.mesh import rasterize_meshes
        from paddle3d.structures import Meshes
    elif type == "standard":
        global standard_rasterize, load_obj
        import os

        from .util import load_obj

        curr_dir = os.path.dirname(__file__)
        standard_rasterize_cuda = paddle.utils.cpp_extension.load(
            name="standard_rasterize_cuda",
            sources=[
                f"{curr_dir}/rasterizer/standard_rasterize_cuda.cpp",
                f"{curr_dir}/rasterizer/standard_rasterize_cuda_kernel.cu",
            ],
            extra_cuda_cflags=["-std=c++14", "-ccbin=$$(which gcc-7)"],
        )
        from standard_rasterize_cuda import standard_rasterize


class StandardRasterizer(paddle.nn.Layer):
    """Alg: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation
    Notice:
        x,y,z are in image space, normalized to [-1, 1]
        can render non-squared image
        not differentiable
    """

    def __init__(self, height, width=None):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        if width is None:
            width = height
        self.h = h = height
        self.w = w = width

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        device = vertices.place
        if h is None:
            h = self.h
        if w is None:
            w = self.h
        bz = vertices.shape[0]
        depth_buffer = (
            paddle.zeros(shape=[bz, h, w]).astype(dtype="float32").to(device)
            + 1000000.0
        )
        triangle_buffer = (
            paddle.zeros(shape=[bz, h, w]).astype(dtype="int32").to(device) - 1
        )
        baryw_buffer = (
            paddle.zeros(shape=[bz, h, w, 3]).astype(dtype="float32").to(device)
        )
        vert_vis = (
            paddle.zeros(shape=[bz, vertices.shape[1]])
            .astype(dtype="float32")
            .to(device)
        )
        vertices = vertices.clone().astype(dtype="float32")
        vertices[(...), :2] = -vertices[(...), :2]
        vertices[..., 0] = vertices[..., 0] * w / 2 + w / 2
        vertices[..., 1] = vertices[..., 1] * h / 2 + h / 2
        vertices[..., 0] = w - 1 - vertices[..., 0]
        vertices[..., 1] = h - 1 - vertices[..., 1]
        vertices[..., 0] = -1 + (2 * vertices[..., 0] + 1) / w
        vertices[..., 1] = -1 + (2 * vertices[..., 1] + 1) / h
        vertices = vertices.clone().astype(dtype="float32")
        vertices[..., 0] = vertices[..., 0] * w / 2 + w / 2
        vertices[..., 1] = vertices[..., 1] * h / 2 + h / 2
        vertices[..., 2] = vertices[..., 2] * w / 2
        f_vs = util.face_vertices(vertices, faces)
        standard_rasterize(f_vs, depth_buffer, triangle_buffer, baryw_buffer, h, w)
        pix_to_face = triangle_buffer[:, :, :, (None)].astype(dtype="int64")
        bary_coords = baryw_buffer[:, :, :, (None), :]
        vismask = (pix_to_face > -1).astype(dtype="float32")
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.reshape(
            (attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        )
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.reshape((N * H * W * K, 1, 1)).expand(
            shape=[N * H * W * K, 3, D]
        )
        pixel_face_vals = attributes.take_along_axis(axis=0, indices=idx).reshape(
            (N, H, W, K, 3, D)
        )
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(axis=-2)
        pixel_vals[mask] = 0
        pixel_vals = pixel_vals[:, :, :, (0)].transpose(perm=[0, 3, 1, 2])
        pixel_vals = paddle.concat(
            x=[pixel_vals, vismask[:, :, :, (0)][:, (None), :, :]], axis=1
        )
        return pixel_vals


class Pytorch3dRasterizer(paddle.nn.Layer):
    """Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            "image_size": image_size,
            "blur_radius": 0.0,
            "faces_per_pixel": 1,
            "bin_size": None,
            "max_faces_per_bin": None,
            "perspective_correct": False,
        }
        raster_settings = util.dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(
        self, vertices, faces, attributes=None, h=None, w=None, return_bary=False
    ):
        fixed_vertices = vertices.clone()
        fixed_vertices[(...), :2] = -fixed_vertices[(...), :2]
        raster_settings = self.raster_settings
        if h is None and w is None:
            image_size = raster_settings.image_size
        else:
            image_size = [h, w]
            if h > w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1] * h / w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0] * w / h
        meshes_screen = Meshes(
            verts=fixed_vertices.astype(dtype="float32"),
            faces=faces.astype(dtype="int64"),
        )
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        vismask = (pix_to_face > -1).astype(dtype="float32")
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.reshape(
            (attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        )
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.reshape((N * H * W * K, 1, 1)).expand(
            shape=[N * H * W * K, 3, D]
        )
        pixel_face_vals = attributes.take_along_axis(axis=0, indices=idx).reshape(
            (N, H, W, K, 3, D)
        )
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(axis=-2)
        pixel_vals[mask] = 0
        pixel_vals = pixel_vals[:, :, :, (0)].transpose(perm=[0, 3, 1, 2])
        pixel_vals = paddle.concat(
            x=[pixel_vals, vismask[:, :, :, (0)][:, (None), :, :]], axis=1
        )
        if return_bary:
            return pixel_vals, bary_coords, pix_to_face
        else:
            return pixel_vals


class SRenderY(paddle.nn.Layer):
    def __init__(
        self, image_size, obj_filename, uv_size=256, rasterizer_type="pytorch3d"
    ):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size
        if rasterizer_type == "pytorch3d":
            self.rasterizer = Pytorch3dRasterizer(image_size)
            self.uv_rasterizer = Pytorch3dRasterizer(uv_size)
            verts, faces, aux = load_obj(obj_filename)
            uvcoords = aux.verts_uvs[None, ...]
            uvfaces = faces.textures_idx[None, ...]
            faces = faces.verts_idx[None, ...]
        elif rasterizer_type == "standard":
            self.rasterizer = StandardRasterizer(image_size)
            self.uv_rasterizer = StandardRasterizer(uv_size)
            verts, uvcoords, faces, uvfaces = load_obj(obj_filename)
            verts = verts[None, ...]
            uvcoords = uvcoords[None, ...]
            faces = faces[None, ...]
            uvfaces = uvfaces[None, ...]
        else:
            NotImplementedError
        dense_triangles = util.generate_triangles(uv_size, uv_size)
        self.register_buffer(
            name="dense_faces",
            tensor=paddle.to_tensor(data=dense_triangles).astype(dtype="int64")[
                (None), :, :
            ],
        )
        self.register_buffer(name="faces", tensor=faces)
        self.register_buffer(name="raw_uvcoords", tensor=uvcoords)
        uvcoords = paddle.concat(x=[uvcoords, uvcoords[:, :, 0:1] * 0.0 + 1.0], axis=-1)
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)
        self.register_buffer(name="uvcoords", tensor=uvcoords)
        self.register_buffer(name="uvfaces", tensor=uvfaces)
        self.register_buffer(name="face_uvcoords", tensor=face_uvcoords)
        colors = (
            paddle.to_tensor(data=[180, 180, 180])[(None), (None), :]
            .tile(repeat_times=[1, faces.max() + 1, 1])
            .astype(dtype="float32")
            / 255.0
        )
        face_colors = util.face_vertices(colors, faces)
        self.register_buffer(name="face_colors", tensor=face_colors)
        pi = np.pi
        constant_factor = paddle.to_tensor(
            data=[
                1 / np.sqrt(4 * pi),
                2 * pi / 3 * np.sqrt(3 / (4 * pi)),
                2 * pi / 3 * np.sqrt(3 / (4 * pi)),
                2 * pi / 3 * np.sqrt(3 / (4 * pi)),
                pi / 4 * 3 * np.sqrt(5 / (12 * pi)),
                pi / 4 * 3 * np.sqrt(5 / (12 * pi)),
                pi / 4 * 3 * np.sqrt(5 / (12 * pi)),
                pi / 4 * (3 / 2) * np.sqrt(5 / (12 * pi)),
                pi / 4 * (1 / 2) * np.sqrt(5 / (4 * pi)),
            ]
        ).astype(dtype="float32")
        self.register_buffer(name="constant_factor", tensor=constant_factor)

    def forward(
        self,
        vertices,
        transformed_vertices,
        albedos,
        lights=None,
        h=None,
        w=None,
        light_type="point",
        background=None,
        add_light=True,
        th=0,
        render_norm=False,
    ):
        """
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], range:normalized to [-1,1], projected vertices in image space (that is aligned to the iamge pixel), for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights:
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        """
        batch_size = vertices.shape[0]
        transformed_vertices[:, :, (2)] = transformed_vertices[:, :, (2)] + 10
        face_vertices = util.face_vertices(
            vertices, self.faces.expand(shape=[batch_size, -1, -1])
        )
        normals = util.vertex_normals(
            vertices, self.faces.expand(shape=[batch_size, -1, -1])
        )
        face_normals = util.face_vertices(
            normals, self.faces.expand(shape=[batch_size, -1, -1])
        )
        transformed_normals = util.vertex_normals(
            transformed_vertices, self.faces.expand(shape=[batch_size, -1, -1])
        )
        transformed_face_normals = util.face_vertices(
            transformed_normals, self.faces.expand(shape=[batch_size, -1, -1])
        )
        attributes = paddle.concat(
            x=[
                self.face_uvcoords.expand(shape=[batch_size, -1, -1, -1]),
                transformed_face_normals.detach(),
                face_vertices.detach(),
                face_normals,
            ],
            axis=-1,
        )
        rendering, bary_cords, pix_to_face = self.rasterizer(
            transformed_vertices,
            self.faces.expand(shape=[batch_size, -1, -1]),
            attributes,
            h,
            w,
            return_bary=True,
        )
        alpha_images = rendering[:, (-1), :, :][:, (None), :, :].detach()
        uvcoords_images = rendering[:, :3, :, :]
        grid = uvcoords_images.transpose(perm=[0, 2, 3, 1])[:, :, :, :2]
        albedo_images = paddle.nn.functional.grid_sample(
            x=albedos, grid=grid, align_corners=False
        )
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).astype(dtype="float32")
        normal_images = rendering[:, 9:12, :, :]
        if add_light:
            if lights is not None:
                if lights.shape[1] == 9:
                    shading_images = self.add_SHlight(normal_images, lights)
                elif light_type == "point":
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(
                        vertice_images.transpose(perm=[0, 2, 3, 1]).reshape(
                            [batch_size, -1, 3]
                        ),
                        normal_images.transpose(perm=[0, 2, 3, 1]).reshape(
                            [batch_size, -1, 3]
                        ),
                        lights,
                        render_norm=render_norm,
                    )
                    shading_images = shading.reshape(
                        [batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]
                    ).transpose(perm=[0, 3, 1, 2])
                else:
                    shading = self.add_directionlight(
                        normal_images.transpose(perm=[0, 2, 3, 1]).reshape(
                            [batch_size, -1, 3]
                        ),
                        -lights,
                    )
                    shading_images = shading.reshape(
                        [batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]
                    ).transpose(perm=[0, 3, 1, 2])
                images = albedo_images * shading_images
            else:
                images = albedo_images
                shading_images = images.detach() * 0.0
        else:
            images = albedo_images
        if background is not None:
            images = images * alpha_images + background * (1.0 - alpha_images)
            albedo_images = albedo_images * alpha_images + background * (
                1.0 - alpha_images
            )
        else:
            images = images * alpha_images
            albedo_images = albedo_images * alpha_images
        outputs = {
            "images": images,
            "albedo_images": albedo_images,
            "alpha_images": alpha_images,
            "pos_mask": pos_mask,
            "shading_images": shading_images,
            "grid": grid,
            "normals": normals,
            "normal_images": normal_images * alpha_images,
            "transformed_normals": transformed_normals,
        }
        return outputs

    def add_SHlight(self, normal_images, sh_coeff):
        """
        sh_coeff: [bz, 9, 3]
        """
        N = normal_images
        sh = paddle.stack(
            x=[
                N[:, (0)] * 0.0 + 1.0,
                N[:, (0)],
                N[:, (1)],
                N[:, (2)],
                N[:, (0)] * N[:, (1)],
                N[:, (0)] * N[:, (2)],
                N[:, (1)] * N[:, (2)],
                N[:, (0)] ** 2 - N[:, (1)] ** 2,
                3 * N[:, (2)] ** 2 - 1,
            ],
            axis=1,
        )
        sh = sh * self.constant_factor[(None), :, (None), (None)]
        shading = paddle.sum(
            x=sh_coeff[:, :, :, (None), (None)] * sh[:, :, (None), :, :], axis=1
        )
        return shading

    def add_pointlight(self, vertices, normals, lights, render_norm=False):
        """
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        """
        light_positions = lights[:, :, :3]
        light_intensities = lights[:, :, 3:]
        directions_to_lights = paddle.nn.functional.normalize(
            x=light_positions[:, :, (None), :] - vertices[:, (None), :, :], axis=3
        )
        normals_dot_lights = (normals[:, (None), :, :] * directions_to_lights).sum(
            axis=3
        )
        shading = (
            normals_dot_lights[:, :, :, (None)] * light_intensities[:, :, (None), :]
        )
        if not render_norm:
            return shading.mean(axis=1)
        else:
            shading = paddle.clip(x=shading, min=0, max=1)
            shading = shading.sum(axis=1)
            shading = shading / shading.amax(axis=(1, 2))
            return shading

    def add_directionlight(self, normals, lights):
        """
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        """
        light_direction = lights[:, :, :3]
        light_intensities = lights[:, :, 3:]
        directions_to_lights = paddle.nn.functional.normalize(
            x=light_direction[:, :, (None), :].expand(
                shape=[-1, -1, normals.shape[1], -1]
            ),
            axis=3,
        )
        normals_dot_lights = paddle.clip(
            x=(normals[:, (None), :, :] * directions_to_lights).sum(axis=3),
            min=0.0,
            max=1.0,
        )
        shading = (
            normals_dot_lights[:, :, :, (None)] * light_intensities[:, :, (None), :]
        )
        return shading.mean(axis=1)

    def render_shape(
        self,
        vertices,
        transformed_vertices,
        colors=None,
        images=None,
        detail_normal_images=None,
        lights=None,
        return_grid=False,
        uv_detail_normals=None,
        h=None,
        w=None,
        threshold=0.15,
    ):
        """
        -- rendering shape with detail normal map
        """
        batch_size = vertices.shape[0]
        if lights is None:
            light_positions = (
                paddle.to_tensor(
                    data=[[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1], [0, 0, 1]]
                )[(None), :, :]
                .expand(shape=[batch_size, -1, -1])
                .astype(dtype="float32")
            )
            light_intensities = (
                paddle.ones_like(x=light_positions).astype(dtype="float32") * 1.7
            )
            lights = paddle.concat(x=(light_positions, light_intensities), axis=2).to(
                vertices.place
            )
        transformed_vertices[:, :, (2)] = transformed_vertices[:, :, (2)] + 10
        face_vertices = util.face_vertices(
            vertices, self.faces.expand(shape=[batch_size, -1, -1])
        )
        normals = util.vertex_normals(
            vertices, self.faces.expand(shape=[batch_size, -1, -1])
        )
        face_normals = util.face_vertices(
            normals, self.faces.expand(shape=[batch_size, -1, -1])
        )
        transformed_normals = util.vertex_normals(
            transformed_vertices, self.faces.expand(shape=[batch_size, -1, -1])
        )
        transformed_face_normals = util.face_vertices(
            transformed_normals, self.faces.expand(shape=[batch_size, -1, -1])
        )
        if colors is None:
            colors = self.face_colors.expand(shape=[batch_size, -1, -1, -1])
        attributes = paddle.concat(
            x=[
                colors,
                transformed_face_normals.detach(),
                face_vertices.detach(),
                face_normals,
                self.face_uvcoords.expand(shape=[batch_size, -1, -1, -1]),
            ],
            axis=-1,
        )
        rendering = self.rasterizer(
            transformed_vertices,
            self.faces.expand(shape=[batch_size, -1, -1]),
            attributes,
            h,
            w,
        )
        alpha_images = rendering[:, (-1), :, :][:, (None), :, :].detach()
        albedo_images = rendering[:, :3, :, :]
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < threshold).astype(
            dtype="float32"
        )
        normal_images = rendering[:, 9:12, :, :].detach()
        vertice_images = rendering[:, 6:9, :, :].detach()
        if detail_normal_images is not None:
            normal_images = detail_normal_images
        shading = self.add_directionlight(
            normal_images.transpose(perm=[0, 2, 3, 1]).reshape([batch_size, -1, 3]),
            lights,
        )
        shading_images = shading.reshape(
            [batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]
        ).transpose(perm=[0, 3, 1, 2])
        shaded_images = albedo_images * shading_images
        alpha_images = alpha_images * pos_mask
        if images is None:
            shape_images = shaded_images * alpha_images + paddle.zeros_like(
                x=shaded_images
            ).to(vertices.place) * (1 - alpha_images)
        else:
            shape_images = shaded_images * alpha_images + images * (1 - alpha_images)
        if return_grid:
            uvcoords_images = rendering[:, 12:15, :, :]
            grid = uvcoords_images.transpose(perm=[0, 2, 3, 1])[:, :, :, :2]
            return (shape_images, normal_images, grid, alpha_images, albedo_images)
        else:
            return shape_images

    def render_depth(self, transformed_vertices):
        """
        -- rendering depth
        """
        batch_size = transformed_vertices.shape[0]
        transformed_vertices[:, :, (2)] = (
            transformed_vertices[:, :, (2)] - transformed_vertices[:, :, (2)].min()
        )
        z = -transformed_vertices[:, :, 2:].tile(repeat_times=[1, 1, 3]).clone()
        z = z - z.min()
        z = z / z.max()
        attributes = util.face_vertices(
            z, self.faces.expand(shape=[batch_size, -1, -1])
        )
        transformed_vertices[:, :, (2)] = transformed_vertices[:, :, (2)] + 10
        rendering = self.rasterizer(
            transformed_vertices,
            self.faces.expand(shape=[batch_size, -1, -1]),
            attributes,
        )
        alpha_images = rendering[:, (-1), :, :][:, (None), :, :].detach()
        depth_images = rendering[:, :1, :, :]
        return depth_images

    def render_colors(self, transformed_vertices, colors):
        """
        -- rendering colors: could be rgb color/ normals, etc
            colors: [bz, num of vertices, 3]
        """
        batch_size = colors.shape[0]
        attributes = util.face_vertices(
            colors, self.faces.expand(shape=[batch_size, -1, -1])
        )
        rendering = self.rasterizer(
            transformed_vertices,
            self.faces.expand(shape=[batch_size, -1, -1]),
            attributes,
        )
        alpha_images = rendering[:, ([-1]), :, :].detach()
        images = rendering[:, :3, :, :] * alpha_images
        return images

    def world2uv(self, vertices):
        """
        warp vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        """
        batch_size = vertices.shape[0]
        face_vertices = util.face_vertices(
            vertices, self.faces.expand(shape=[batch_size, -1, -1])
        )
        uv_vertices = self.uv_rasterizer(
            self.uvcoords.expand(shape=[batch_size, -1, -1]),
            self.uvfaces.expand(shape=[batch_size, -1, -1]),
            face_vertices,
        )[:, :3]
        return uv_vertices
