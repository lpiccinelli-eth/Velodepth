import torch
from torch.nn import functional as F


def quaternion_to_R(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def R_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    out = standardize_quaternion(out)
    return out


def Rt_to_pose(R, t):
    assert R.shape[-2:] == (3, 3), "The last two dimensions of R must be 3x3"
    assert t.shape[-2:] == (3, 1), "The last dimension of t must be 3"

    # Create the pose matrix
    pose = torch.cat([R, t], dim=-1)
    pose = F.pad(pose, (0, 0, 0, 1), value=0)
    pose[..., 3, 3] = 1

    return pose


def pose_to_Rt(pose):
    assert pose.shape[-2:] == (4, 4), "The last two dimensions of pose must be 4x4"

    # Extract the rotation matrix and translation vector
    R = pose[..., :3, :3]
    t = pose[..., :3, 3:]

    return R, t


def relative_pose(pose1, pose2):
    # Compute world_to_cam for pose1
    pose1_inv = invert_pose(pose1)

    # Relative pose as cam_to_world_2 -> world_to_cam_1 => cam2_to_cam1
    relative_pose = pose1_inv @ pose2

    return relative_pose


@torch.autocast(device_type="cuda", dtype=torch.float32)
def invert_pose(pose):
    R, t = pose_to_Rt(pose)
    R_inv = R.transpose(-2, -1)
    t_inv = -torch.matmul(R_inv, t)
    pose_inv = Rt_to_pose(R_inv, t_inv)
    return pose_inv


def apply_pose_transformation(point_cloud, pose):
    reshape = point_cloud.ndim > 3
    shapes = point_cloud.shape
    # Extract rotation and translation from pose
    R, t = pose_to_Rt(pose)

    # Apply the pose transformation
    if reshape:
        point_cloud = point_cloud.reshape(shapes[0], -1, shapes[-1])
    transformed_points = torch.matmul(point_cloud, R.transpose(-2, -1)) + t.transpose(
        -2, -1
    )
    if reshape:
        transformed_points = transformed_points.reshape(shapes)
    return transformed_points


def euler2mat(roll, pitch, yaw) -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.

    Args:
        euler_angles (torch.Tensor): Tensor of shape (N, 3) representing roll, pitch, yaw in radians.
            - roll: rotation around z-axis
            - pitch: rotation around x-axis
            - yaw: rotation around y-axis
    Returns:
        torch.Tensor: Tensor of shape (N, 3, 3) representing the rotation matrices.
    """

    cos_r, sin_r = torch.cos(roll), torch.sin(roll)  # Roll
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)  # Pitch
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)  # Yaw

    # Rotation matrices
    R_z = torch.zeros((roll.shape[0], 3, 3), device=roll.device)
    R_y = torch.zeros_like(R_z)
    R_x = torch.zeros_like(R_z)

    # Z-axis (roll)
    R_z[:, 0, 0], R_z[:, 0, 1], R_z[:, 1, 0], R_z[:, 1, 1], R_z[:, 2, 2] = (
        cos_y,
        -sin_y,
        sin_y,
        cos_y,
        1.0,
    )

    # Y-axis (yaw)
    R_y[:, 0, 0], R_y[:, 0, 2], R_y[:, 2, 0], R_y[:, 2, 2], R_y[:, 1, 1] = (
        cos_p,
        sin_p,
        -sin_p,
        cos_p,
        1.0,
    )

    # X-axis (pitch)
    R_x[:, 1, 1], R_x[:, 1, 2], R_x[:, 2, 1], R_x[:, 2, 2], R_x[:, 0, 0] = (
        cos_r,
        -sin_r,
        sin_r,
        cos_r,
        1.0,
    )

    # Combine rotations: R = R_z * R_y * R_x
    rotation_matrix = torch.matmul(torch.matmul(R_z, R_y), R_x)
    return rotation_matrix
