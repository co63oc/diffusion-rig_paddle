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
import numpy as np
import paddle

import utils.paddle_add


def rot_mat_to_euler(rot_mats):
    sy = paddle.sqrt(
        x=rot_mats[:, (0), (0)] * rot_mats[:, (0), (0)]
        + rot_mats[:, (1), (0)] * rot_mats[:, (1), (0)]
    )
    return paddle.atan2(x=-rot_mats[:, (2), (0)], y=sy)


def find_dynamic_lmk_idx_and_bcoords(
    vertices,
    pose,
    dynamic_lmk_faces_idx,
    dynamic_lmk_b_coords,
    neck_kin_chain,
    dtype="float32",
):
    """Compute the faces, barycentric coordinates for the dynamic landmarks


    To do so, we first compute the rotation of the neck around the y-axis
    and then use a pre-computed look-up table to find the faces and the
    barycentric coordinates that will be used.

    Special thanks to Soubhik Sanyal (soubhik.sanyal@tuebingen.mpg.de)
    for providing the original TensorFlow implementation and for the LUT.

    Parameters
    ----------
    vertices: torch.tensor BxVx3, dtype = torch.float32
        The tensor of input vertices
    pose: torch.tensor Bx(Jx3), dtype = torch.float32
        The current pose of the body model
    dynamic_lmk_faces_idx: torch.tensor L, dtype = torch.long
        The look-up table from neck rotation to faces
    dynamic_lmk_b_coords: torch.tensor Lx3, dtype = torch.float32
        The look-up table from neck rotation to barycentric coordinates
    neck_kin_chain: list
        A python list that contains the indices of the joints that form the
        kinematic chain of the neck.
    dtype: torch.dtype, optional

    Returns
    -------
    dyn_lmk_faces_idx: torch.tensor, dtype = torch.long
        A tensor of size BxL that contains the indices of the faces that
        will be used to compute the current dynamic landmarks.
    dyn_lmk_b_coords: torch.tensor, dtype = torch.float32
        A tensor of size BxL that contains the indices of the faces that
        will be used to compute the current dynamic landmarks.
    """
    batch_size = vertices.shape[0]
    aa_pose = paddle.index_select(
        x=pose.reshape([batch_size, -1, 3]), axis=1, index=neck_kin_chain
    )
    rot_mats = batch_rodrigues(aa_pose.reshape([-1, 3]), dtype=dtype).reshape(
        [batch_size, -1, 3, 3]
    )
    rel_rot_mat = paddle.eye(num_rows=3).astype(dtype).unsqueeze_(axis=0)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = paddle.bmm(x=rot_mats[:, (idx)], y=rel_rot_mat)
    y_rot_angle = paddle.round(
        paddle.clip(x=-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
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


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    """Calculates landmarks by barycentric interpolation

    Parameters
    ----------
    vertices: torch.tensor BxVx3, dtype = torch.float32
        The tensor of input vertices
    faces: torch.tensor Fx3, dtype = torch.long
        The faces of the mesh
    lmk_faces_idx: torch.tensor L, dtype = torch.long
        The tensor with the indices of the faces used to calculate the
        landmarks.
    lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
        The tensor of barycentric coordinates that are used to interpolate
        the landmarks

    Returns
    -------
    landmarks: torch.tensor BxLx3, dtype = torch.float32
        The coordinates of the landmarks for each mesh in the batch
    """
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.place
    lmk_faces = paddle.index_select(
        x=faces, axis=0, index=lmk_faces_idx.reshape([-1])
    ).reshape([batch_size, -1, 3])
    lmk_faces += (
        paddle.arange(end=batch_size).astype("int64").reshape([-1, 1, 1]) * num_verts
    )
    lmk_vertices = vertices.reshape([-1, 3])[lmk_faces].reshape([batch_size, -1, 3, 3])
    landmarks = paddle.einsum("blfi,blf->bli", [lmk_vertices, lmk_bary_coords])
    return landmarks


def lbs(
    betas,
    pose,
    v_template,
    shapedirs,
    posedirs,
    J_regressor,
    parents,
    lbs_weights,
    pose2rot=True,
    dtype="float32",
):
    """Performs Linear Blend Skinning with the given shape and pose parameters

    Parameters
    ----------
    betas : torch.tensor BxNB
        The tensor of shape parameters
    pose : torch.tensor Bx(J + 1) * 3
        The pose parameters in axis-angle format
    v_template torch.tensor BxVx3
        The template mesh that will be deformed
    shapedirs : torch.tensor 1xNB
        The tensor of PCA shape displacements
    posedirs : torch.tensor Px(V * 3)
        The pose PCA coefficients
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from
        the position of the vertices
    parents: torch.tensor J
        The array that describes the kinematic tree for the model
    lbs_weights: torch.tensor N x V x (J + 1)
        The linear blend skinning weights that represent how much the
        rotation matrix of each part affects each vertex
    pose2rot: bool, optional
        Flag on whether to convert the input pose tensor to rotation
        matrices. The default value is True. If False, then the pose tensor
        should already contain rotation matrices and have a size of
        Bx(J + 1)x9
    dtype: torch.dtype, optional

    Returns
    -------
    verts: torch.tensor BxVx3
        The vertices of the mesh after applying the shape and pose
        displacements.
    joints: torch.tensor BxJx3
        The joints of the model
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.place
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)
    ident = paddle.eye(num_rows=3).astype(dtype)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.reshape([-1, 3]), dtype=dtype).reshape(
            [batch_size, -1, 3, 3]
        )
        pose_feature = (rot_mats[:, 1:, :, :] - ident).reshape([batch_size, -1])
        pose_offsets = paddle.matmul(x=pose_feature, y=posedirs).reshape(
            [batch_size, -1, 3]
        )
    else:
        pose_feature = pose[:, 1:].reshape([batch_size, -1, 3, 3]) - ident
        rot_mats = pose.reshape([batch_size, -1, 3, 3])
        pose_offsets = paddle.matmul(
            x=pose_feature.reshape([batch_size, -1]), y=posedirs
        ).reshape([batch_size, -1, 3])
    v_posed = pose_offsets + v_shaped
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    W = lbs_weights.unsqueeze(axis=0).expand(shape=[batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = paddle.matmul(x=W, y=A.reshape([batch_size, num_joints, 16])).reshape(
        [batch_size, -1, 4, 4]
    )
    homogen_coord = paddle.ones(shape=[batch_size, v_posed.shape[1], 1], dtype=dtype)
    v_posed_homo = paddle.concat(x=[v_posed, homogen_coord], axis=2)
    v_homo = paddle.matmul(x=T, y=paddle.unsqueeze(x=v_posed_homo, axis=-1))
    verts = v_homo[:, :, :3, (0)]
    return verts, J_transformed


def vertices2joints(J_regressor, vertices):
    """Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    """
    return paddle.einsum("bik,ji->bjk", [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    """Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """
    blend_shape = paddle.einsum("bl,mkl->bmk", [betas, shape_disps])
    return blend_shape


def batch_rodrigues(rot_vecs, epsilon=1e-08, dtype="float32"):
    """Calculates the rotation matrices for a batch of rotation vectors
    Parameters
    ----------
    rot_vecs: torch.tensor Nx3
        array of N axis-angle vectors
    Returns
    -------
    R: torch.tensor Nx3x3
        The rotation matrices for the given axis-angle parameters
    """
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.place
    angle = paddle.linalg.norm(x=rot_vecs + 1e-08, axis=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = paddle.unsqueeze(x=paddle.cos(x=angle), axis=1)
    sin = paddle.unsqueeze(x=paddle.sin(x=angle), axis=1)
    rx, ry, rz = paddle_add.split(
        x=rot_dir, num_or_sections=rot_dir.shape[1] // 1, axis=1
    )
    K = paddle.zeros(shape=(batch_size, 3, 3), dtype=dtype)
    zeros = paddle.zeros(shape=(batch_size, 1), dtype=dtype)
    K = paddle.concat(
        x=[zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1
    ).reshape((batch_size, 3, 3))
    ident = paddle.eye(num_rows=3).astype(dtype).unsqueeze(axis=0)
    rot_mat = ident + sin * K + (1 - cos) * paddle.bmm(x=K, y=K)
    return rot_mat


def transform_mat(R, t):
    """Creates a batch of transformation matrices
    Args:
        - R: Bx3x3 array of a batch of rotation matrices
        - t: Bx3x1 array of a batch of translation vectors
    Returns:
        - T: Bx4x4 Transformation matrix
    """
    return paddle.concat(
        x=[paddle_add.pad(R, [0, 0, 0, 1]), paddle_add.pad(t, [0, 0, 0, 1], value=1)],
        axis=2,
    )


def batch_rigid_transform(rot_mats, joints, parents, dtype="float32"):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = paddle.unsqueeze(x=joints, axis=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, (parents[1:])]
    transforms_mat = transform_mat(
        rot_mats.reshape([-1, 3, 3]), rel_joints.reshape(-1, 3, 1)
    ).reshape((-1, joints.shape[1], 4, 4))
    transform_chain = [transforms_mat[:, (0)]]
    for i in range(1, parents.shape[0]):
        curr_res = paddle.matmul(
            x=transform_chain[parents[i]], y=transforms_mat[:, (i)]
        )
        transform_chain.append(curr_res)
    transforms = paddle.stack(x=transform_chain, axis=1)
    posed_joints = transforms[:, :, :3, (3)]
    posed_joints = transforms[:, :, :3, (3)]
    joints_homogen = paddle_add.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - paddle_add.pad(
        paddle.matmul(x=transforms, y=joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
    )
    return posed_joints, rel_transforms
