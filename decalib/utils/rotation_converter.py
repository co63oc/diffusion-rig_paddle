import sys
sys.path.append('/nfs/github/recurrent/out/utils')
import paddle_aux
import paddle
""" Rotation Converter
Repre: euler angle(3), angle axis(3), rotation matrix(3x3), quaternion(4)
ref: https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/core/conversions.html#
"pi",
"rad2deg",
"deg2rad",
# "angle_axis_to_rotation_matrix", batch_rodrigues
"rotation_matrix_to_angle_axis",
"rotation_matrix_to_quaternion",
"quaternion_to_angle_axis",
# "angle_axis_to_quaternion",

euler2quat_conversion_sanity_batch

ref: smplx/lbs
batch_rodrigues: axis angle -> matrix
# 
"""
pi = paddle.to_tensor(data=[3.141592653589793], dtype='float32')


def rad2deg(tensor):
    """Function that converts angles from radians to degrees.

    See :class:`~torchgeometry.RadToDeg` for details.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Example:
        >>> input = tgm.pi * torch.rand(1, 3, 3)
        >>> output = tgm.rad2deg(input)
    """
    if not paddle.is_tensor(x=tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(tensor)))
    return 180.0 * tensor / pi.to(tensor.place).astype(tensor.dtype)


def deg2rad(tensor):
    """Function that converts angles from degrees to radians.

    See :class:`~torchgeometry.DegToRad` for details.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Examples::

        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = tgm.deg2rad(input)
    """
    if not paddle.is_tensor(x=tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(tensor)))
    return tensor * pi.to(tensor.place).astype(tensor.dtype) / 180.0


def euler_to_quaternion(r):
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]
    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = paddle.cos(x=z)
    sz = paddle.sin(x=z)
    cy = paddle.cos(x=y)
    sy = paddle.sin(x=y)
    cx = paddle.cos(x=x)
    sx = paddle.sin(x=x)
    quaternion = paddle.zeros_like(x=r.tile(repeat_times=[1, 2]))[(...), :4
        ].to(r.place)
    quaternion[..., 0] += cx * cy * cz - sx * sy * sz
    quaternion[..., 1] += cx * sy * sz + cy * cz * sx
    quaternion[..., 2] += cx * cz * sy - sx * cy * sz
    quaternion[..., 3] += cx * cy * sz + sx * cz * sy
    return quaternion


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-06):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not paddle.is_tensor(x=rotation_matrix):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(rotation_matrix)))
    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            'Input size must be a three dimensional tensor. Got {}'.format(
            rotation_matrix.shape))
    x = rotation_matrix
    perm_0 = list(range(x.ndim))
    perm_0[1] = 2
    perm_0[2] = 1
    rmat_t = paddle.transpose(x=x, perm=perm_0)
    mask_d2 = rmat_t[:, (2), (2)] < eps
    mask_d0_d1 = rmat_t[:, (0), (0)] > rmat_t[:, (1), (1)]
    mask_d0_nd1 = rmat_t[:, (0), (0)] < -rmat_t[:, (1), (1)]
    t0 = 1 + rmat_t[:, (0), (0)] - rmat_t[:, (1), (1)] - rmat_t[:, (2), (2)]
    q0 = paddle.stack(x=[rmat_t[:, (1), (2)] - rmat_t[:, (2), (1)], t0, 
        rmat_t[:, (0), (1)] + rmat_t[:, (1), (0)], rmat_t[:, (2), (0)] +
        rmat_t[:, (0), (2)]], axis=-1)
    t0_rep = t0.tile(repeat_times=[4, 1]).t()
    t1 = 1 - rmat_t[:, (0), (0)] + rmat_t[:, (1), (1)] - rmat_t[:, (2), (2)]
    q1 = paddle.stack(x=[rmat_t[:, (2), (0)] - rmat_t[:, (0), (2)], rmat_t[
        :, (0), (1)] + rmat_t[:, (1), (0)], t1, rmat_t[:, (1), (2)] +
        rmat_t[:, (2), (1)]], axis=-1)
    t1_rep = t1.tile(repeat_times=[4, 1]).t()
    t2 = 1 - rmat_t[:, (0), (0)] - rmat_t[:, (1), (1)] + rmat_t[:, (2), (2)]
    q2 = paddle.stack(x=[rmat_t[:, (0), (1)] - rmat_t[:, (1), (0)], rmat_t[
        :, (2), (0)] + rmat_t[:, (0), (2)], rmat_t[:, (1), (2)] + rmat_t[:,
        (2), (1)], t2], axis=-1)
    t2_rep = t2.tile(repeat_times=[4, 1]).t()
    t3 = 1 + rmat_t[:, (0), (0)] + rmat_t[:, (1), (1)] + rmat_t[:, (2), (2)]
    q3 = paddle.stack(x=[t3, rmat_t[:, (1), (2)] - rmat_t[:, (2), (1)], 
        rmat_t[:, (2), (0)] - rmat_t[:, (0), (2)], rmat_t[:, (0), (1)] -
        rmat_t[:, (1), (0)]], axis=-1)
    t3_rep = t3.tile(repeat_times=[4, 1]).t()
    mask_c0 = mask_d2 * mask_d0_d1.astype(dtype='float32')
    mask_c1 = mask_d2 * (1 - mask_d0_d1.astype(dtype='float32'))
    mask_c2 = (1 - mask_d2.astype(dtype='float32')) * mask_d0_nd1
    mask_c3 = (1 - mask_d2.astype(dtype='float32')) * (1 - mask_d0_nd1.
        astype(dtype='float32'))
    mask_c0 = mask_c0.reshape([-1, 1]).astype(dtype=q0.dtype)
    mask_c1 = mask_c1.reshape([-1, 1]).astype(dtype=q1.dtype)
    mask_c2 = mask_c2.reshape([-1, 1]).astype(dtype=q2.dtype)
    mask_c3 = mask_c3.reshape([-1, 1]).astype(dtype=q3.dtype)
    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= paddle.sqrt(x=t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep *
        mask_c2 + t3_rep * mask_c3)
    q *= 0.5
    return q


def angle_axis_to_quaternion(angle_axis: paddle.Tensor) ->paddle.Tensor:
    """Convert an angle axis to a quaternion.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        angle_axis (torch.Tensor): tensor with angle axis.

    Return:
        torch.Tensor: tensor with quaternion.

    Shape:
        - Input: :math:`(*, 3)` where `*` means, any number of dimensions
        - Output: :math:`(*, 4)`

    Example:
        >>> angle_axis = torch.rand(2, 4)  # Nx4
        >>> quaternion = tgm.angle_axis_to_quaternion(angle_axis)  # Nx3
    """
    if not paddle.is_tensor(x=angle_axis):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(angle_axis)))
    if not angle_axis.shape[-1] == 3:
        raise ValueError('Input must be a tensor of shape Nx3 or 3. Got {}'
            .format(angle_axis.shape))
    a0: paddle.Tensor = angle_axis[(...), 0:1]
    a1: paddle.Tensor = angle_axis[(...), 1:2]
    a2: paddle.Tensor = angle_axis[(...), 2:3]
    theta_squared: paddle.Tensor = a0 * a0 + a1 * a1 + a2 * a2
    theta: paddle.Tensor = paddle.sqrt(x=theta_squared)
    half_theta: paddle.Tensor = theta * 0.5
    mask: paddle.Tensor = theta_squared > 0.0
    ones: paddle.Tensor = paddle.ones_like(x=half_theta)
    k_neg: paddle.Tensor = 0.5 * ones
    k_pos: paddle.Tensor = paddle.sin(x=half_theta) / theta
    k: paddle.Tensor = paddle.where(condition=mask, x=k_pos, y=k_neg)
    w: paddle.Tensor = paddle.where(condition=mask, x=paddle.cos(x=
        half_theta), y=ones)
    quaternion: paddle.Tensor = paddle.zeros_like(x=angle_axis)
    quaternion[(...), 0:1] += a0 * k
    quaternion[(...), 1:2] += a1 * k
    quaternion[(...), 2:3] += a2 * k
    return paddle.concat(x=[w, quaternion], axis=-1)


def quaternion_to_rotation_matrix(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, axis=1, keepdim=True)
    w, x, y, z = norm_quat[:, (0)], norm_quat[:, (1)], norm_quat[:, (2)
        ], norm_quat[:, (3)]
    B = quat.shape[0]
    w2, x2, y2, z2 = w.pow(y=2), x.pow(y=2), y.pow(y=2), z.pow(y=2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = paddle.stack(x=[w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 *
        xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 *
        wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], axis=1).reshape([B, 3, 3])
    return rotMat


def quaternion_to_angle_axis(quaternion: paddle.Tensor):
    """Convert quaternion vector to angle axis of rotation. TODO: CORRECT

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not paddle.is_tensor(x=quaternion):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(quaternion)))
    if not quaternion.shape[-1] == 4:
        raise ValueError('Input must be a tensor of shape Nx4 or 4. Got {}'
            .format(quaternion.shape))
    q1: paddle.Tensor = quaternion[..., 1]
    q2: paddle.Tensor = quaternion[..., 2]
    q3: paddle.Tensor = quaternion[..., 3]
    sin_squared_theta: paddle.Tensor = q1 * q1 + q2 * q2 + q3 * q3
    sin_theta: paddle.Tensor = paddle.sqrt(x=sin_squared_theta)
    cos_theta: paddle.Tensor = quaternion[..., 0]
    two_theta: paddle.Tensor = 2.0 * paddle.where(condition=cos_theta < 0.0,
        x=paddle.atan2(x=-sin_theta, y=-cos_theta), y=paddle.atan2(x=
        sin_theta, y=cos_theta))
    k_pos: paddle.Tensor = two_theta / sin_theta
    k_neg: paddle.Tensor = 2.0 * paddle.ones_like(x=sin_theta).to(quaternion
        .place)
    k: paddle.Tensor = paddle.where(condition=sin_squared_theta > 0.0, x=
        k_pos, y=k_neg)
    angle_axis: paddle.Tensor = paddle.zeros_like(x=quaternion).to(quaternion
        .place)[(...), :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def batch_euler2axis(r):
    return quaternion_to_angle_axis(euler_to_quaternion(r))


def batch_euler2matrix(r):
    return quaternion_to_rotation_matrix(euler_to_quaternion(r))


def batch_matrix2euler(rot_mats):
    sy = paddle.sqrt(x=rot_mats[:, (0), (0)] * rot_mats[:, (0), (0)] + 
        rot_mats[:, (1), (0)] * rot_mats[:, (1), (0)])
    return paddle.atan2(x=-rot_mats[:, (2), (0)], y=sy)


def batch_matrix2axis(rot_mats):
    return quaternion_to_angle_axis(rotation_matrix_to_quaternion(rot_mats))


def batch_axis2matrix(theta):
    return quaternion_to_rotation_matrix(angle_axis_to_quaternion(theta))


def batch_axis2euler(theta):
    return batch_matrix2euler(batch_axis2matrix(theta))


def batch_axis2euler(r):
    return rot_mat_to_euler(batch_rodrigues(r))


def batch_orth_proj(X, camera):
    """
        X is N x num_pquaternion_to_angle_axisoints x 3
    """
    camera = camera.clone().reshape([-1, 1, 3])
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = paddle.concat(x=[X_trans, X[:, :, 2:]], axis=2)
    Xn = camera[:, :, 0:1] * X_trans
    return Xn


def batch_rodrigues(rot_vecs, epsilon=1e-08, dtype='float32'):
    """  same as batch_matrix2axis
    Calculates the rotation matrices for a batch of rotation vectors
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
    rx, ry, rz = paddle.split(x=rot_dir, num_or_sections=rot_dir.shape[1] //
        1, axis=1)
    K = paddle.zeros(shape=(batch_size, 3, 3), dtype=dtype)
    zeros = paddle.zeros(shape=(batch_size, 1), dtype=dtype)
    K = paddle.concat(x=[zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
        axis=1).reshape((batch_size, 3, 3))
    ident = paddle.eye(num_rows=3).astype(dtype).unsqueeze(axis=0)
    rot_mat = ident + sin * K + (1 - cos) * paddle.bmm(x=K, y=K)
    return rot_mat
