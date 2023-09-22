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

import pickle

import numpy as np
import paddle

from .lbs import batch_rodrigues, lbs, rot_mat_to_euler, vertices2landmarks


def to_tensor(array, dtype="float32"):
    if "torch.tensor" not in str(type(array)):
        return paddle.to_tensor(data=array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class FLAME(paddle.nn.Layer):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, config):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open(config.flame_model_path, "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)
        self.dtype = "float32"
        self.register_buffer(
            name="faces_tensor",
            tensor=to_tensor(to_np(flame_model.f, dtype=np.int64), dtype="int64"),
        )
        self.register_buffer(
            name="v_template",
            tensor=to_tensor(to_np(flame_model.v_template), dtype=self.dtype),
        )
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = paddle.concat(
            x=[
                shapedirs[:, :, : config.n_shape],
                shapedirs[:, :, 300 : 300 + config.n_exp],
            ],
            axis=2,
        )
        self.register_buffer(name="shapedirs", tensor=shapedirs)
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer(
            name="posedirs", tensor=to_tensor(to_np(posedirs), dtype=self.dtype)
        )
        self.register_buffer(
            name="J_regressor",
            tensor=to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype),
        )
        parents = to_tensor(to_np(flame_model.kintree_table[0])).astype(dtype="int64")
        parents[0] = -1
        self.register_buffer(name="parents", tensor=parents)
        self.register_buffer(
            name="lbs_weights",
            tensor=to_tensor(to_np(flame_model.weights), dtype=self.dtype),
        )
        out_0 = paddle.zeros(shape=[1, 6], dtype=self.dtype)
        out_0.stop_gradient = not False
        default_eyball_pose = out_0
        out_1 = paddle.create_parameter(
            shape=default_eyball_pose.shape,
            dtype=default_eyball_pose.numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(default_eyball_pose),
        )
        out_1.stop_gradient = not False
        self.add_parameter(name="eye_pose", parameter=out_1)
        out_2 = paddle.zeros(shape=[1, 3], dtype=self.dtype)
        out_2.stop_gradient = not False
        default_neck_pose = out_2
        out_3 = paddle.create_parameter(
            shape=default_neck_pose.shape,
            dtype=default_neck_pose.numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(default_neck_pose),
        )
        out_3.stop_gradient = not False
        self.add_parameter(name="neck_pose", parameter=out_3)
        lmk_embeddings = np.load(
            config.flame_lmk_embedding_path, allow_pickle=True, encoding="latin1"
        )
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            name="lmk_faces_idx",
            tensor=paddle.to_tensor(data=lmk_embeddings["static_lmk_faces_idx"]).astype(
                dtype="int64"
            ),
        )
        self.register_buffer(
            name="lmk_bary_coords",
            tensor=paddle.to_tensor(
                data=lmk_embeddings["static_lmk_bary_coords"], dtype=self.dtype
            ),
        )
        self.register_buffer(
            name="dynamic_lmk_faces_idx",
            tensor=paddle.to_tensor(lmk_embeddings["dynamic_lmk_faces_idx"]).astype(
                dtype="int64"
            ),
        )
        self.register_buffer(
            name="dynamic_lmk_bary_coords",
            tensor=paddle.to_tensor(lmk_embeddings["dynamic_lmk_bary_coords"]).astype(
                self.dtype
            ),
        )
        self.register_buffer(
            name="full_lmk_faces_idx",
            tensor=paddle.to_tensor(data=lmk_embeddings["full_lmk_faces_idx"]).astype(
                dtype="int64"
            ),
        )
        self.register_buffer(
            name="full_lmk_bary_coords",
            tensor=paddle.to_tensor(data=lmk_embeddings["full_lmk_bary_coords"]).astype(
                self.dtype
            ),
        )
        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = paddle.to_tensor(data=NECK_IDX, dtype="int64")
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer(
            name="neck_kin_chain", tensor=paddle.stack(x=neck_kin_chain)
        )

    def _find_dynamic_lmk_idx_and_bcoords(
        self,
        pose,
        dynamic_lmk_faces_idx,
        dynamic_lmk_b_coords,
        neck_kin_chain,
        dtype="float32",
    ):
        """
        Selects the face contour depending on the reletive position of the head
        Input:
            vertices: N X num_of_vertices X 3
            pose: N X full pose
            dynamic_lmk_faces_idx: The list of contour face indexes
            dynamic_lmk_b_coords: The list of contour barycentric weights
            neck_kin_chain: The tree to consider for the relative rotation
            dtype: Data type
        return:
            The contour face indexes and the corresponding barycentric weights
        """
        batch_size = pose.shape[0]
        aa_pose = paddle.index_select(
            x=pose.reshape([batch_size, -1, 3]), axis=1, index=neck_kin_chain
        )
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3), dtype=dtype).reshape(
            [batch_size, -1, 3, 3]
        )
        rel_rot_mat = (
            paddle.eye(num_rows=3)
            .astype(dtype)
            .unsqueeze_(axis=0)
            .expand(shape=[batch_size, -1, -1])
        )
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = paddle.bmm(x=rot_mats[:, (idx)], y=rel_rot_mat)
        y_rot_angle = paddle.round(
            paddle.clip(x=rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
        ).astype(dtype="int64")
        neg_mask = y_rot_angle.less_than(y=paddle.to_tensor(0)).astype(dtype="int64")
        mask = y_rot_angle.less_than(y=paddle.to_tensor(-39)).astype(dtype="int64")
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle
        dyn_lmk_faces_idx = paddle.index_select(
            x=dynamic_lmk_faces_idx, axis=0, index=y_rot_angle
        )
        dyn_lmk_b_coords = paddle.index_select(
            x=dynamic_lmk_b_coords, axis=0, index=y_rot_angle
        )
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
        Calculates landmarks by barycentric interpolation
        Input:
            vertices: torch.tensor NxVx3, dtype = torch.float32
                The tensor of input vertices
            faces: torch.tensor (N*F)x3, dtype = torch.long
                The faces of the mesh
            lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                The tensor with the indices of the faces used to calculate the
                landmarks.
            lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                The tensor of barycentric coordinates that are used to interpolate
                the landmarks

        Returns:
            landmarks: torch.tensor NxLx3, dtype = torch.float32
                The coordinates of the landmarks for each mesh in the batch
        """
        batch_size, num_verts = vertices.shape[:dd2]
        lmk_faces = (
            paddle.index_select(x=faces, axis=0, index=lmk_faces_idx.view(-1))
            .view(1, -1, 3)
            .reshape([batch_size, lmk_faces_idx.shape[1], -1])
        )
        lmk_faces += (
            paddle.to_tensor(
                paddle.arange(end=batch_size).astype("int64").reshape([-1, 1, 1]),
                place=vertices.place,
            )
            * num_verts
        )
        lmk_vertices = vertices.reshape([-1, 3])[lmk_faces]
        landmarks = paddle.einsum("blfi,blf->bli", [lmk_vertices, lmk_bary_coords])
        return landmarks

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.tile(repeat_times=[vertices.shape[0], 1]),
            self.full_lmk_bary_coords.tile(repeat_times=[vertices.shape[0], 1, 1]),
        )
        return landmarks3d

    def forward(
        self,
        shape_params=None,
        expression_params=None,
        pose_params=None,
        eye_pose_params=None,
    ):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if pose_params is None:
            pose_params = self.eye_pose.expand(shape=[batch_size, -1])
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(shape=[batch_size, -1])
        betas = paddle.concat(x=[shape_params, expression_params], axis=1)
        full_pose = paddle.concat(
            x=[
                pose_params[:, :3],
                self.neck_pose.expand(shape=[batch_size, -1]),
                pose_params[:, 3:],
                eye_pose_params,
            ],
            axis=1,
        )
        template_vertices = self.v_template.unsqueeze(axis=0).expand(
            shape=[batch_size, -1, -1]
        )
        vertices, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
        )
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(axis=0).expand(
            shape=[batch_size, -1]
        )
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(axis=0).expand(
            shape=[batch_size, -1, -1]
        )
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose,
            self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain,
            dtype=self.dtype,
        )
        lmk_faces_idx = paddle.concat(x=[dyn_lmk_faces_idx, lmk_faces_idx], axis=1)
        lmk_bary_coords = paddle.concat(
            x=[dyn_lmk_bary_coords, lmk_bary_coords], axis=1
        )
        landmarks2d = vertices2landmarks(
            vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords
        )
        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.tile(repeat_times=[bz, 1]),
            self.full_lmk_bary_coords.tile(repeat_times=[bz, 1, 1]),
        )
        return vertices, landmarks2d, landmarks3d


class FLAMETex(paddle.nn.Layer):
    """
    FLAME texture:
    https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
    FLAME texture converted from BFM:
    https://github.com/TimoBolkart/BFM_to_FLAME
    """

    def __init__(self, config):
        super(FLAMETex, self).__init__()
        if config.tex_type == "BFM":
            mu_key = "MU"
            pc_key = "PC"
            n_pc = 199
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape((1, -1))
            texture_basis = tex_space[pc_key].reshape((-1, n_pc))
        elif config.tex_type == "FLAME":
            mu_key = "mean"
            pc_key = "tex_dir"
            n_pc = 200
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape((1, -1)) / 255.0
            texture_basis = tex_space[pc_key].reshape((-1, n_pc)) / 255.0
        else:
            print("texture type ", config.tex_type, "not exist!")
            raise NotImplementedError
        n_tex = config.n_tex
        num_components = texture_basis.shape[1]
        texture_mean = paddle.to_tensor(data=texture_mean).astype(dtype="float32")[
            None, ...
        ]
        texture_basis = paddle.to_tensor(data=texture_basis[:, :n_tex]).astype(
            dtype="float32"
        )[None, ...]
        self.register_buffer(name="texture_mean", tensor=texture_mean)
        self.register_buffer(name="texture_basis", tensor=texture_basis)

    def forward(self, texcode):
        """
        texcode: [batchsize, n_tex]
        texture: [bz, 3, 256, 256], range: 0-1
        """
        texture = self.texture_mean + (self.texture_basis * texcode[:, (None), :]).sum(
            axis=-1
        )
        texture = texture.reshape((texcode.shape[0], 512, 512, 3)).transpose(
            perm=[0, 3, 1, 2]
        )
        texture = paddle.nn.functional.interpolate(x=texture, size=[256, 256])
        texture = texture[:, ([2, 1, 0]), :, :]
        return texture
