"""
Utility functions of matrix and vector transformations.
Based on the utility functions from Robosuite (https://github.com/StanfordVL/robosuite)

NOTE: convention for quaternions is (x, y, z, w)
"""

import dm_control.utils.transformations as T
import math
import numba
import numpy as np
from scipy.spatial.transform import Rotation as R

EPS = np.finfo(np.float32).eps * 4.


def get_normalized_quat(pose):
    if pose.shape[-1] in [7, 8, 10, 11]:
        q = pose[..., 3:7]
    elif pose.shape[-1] == 6:
        q = fast_euler2quat(pose[..., 3:]).astype(pose.dtype)
    elif pose.shape[-1] == 9:
        # rot 6d
        q = mat2quat(rot6d2mat(pose[..., 3:])).astype(pose.dtype)
    else:
        raise NotImplementedError(str(pose.shape))

    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def convert_quat(q, to="xyzw"):
    """
    Converts quaternion from one convention to another.
    The convention to convert TO is specified as an optional argument.
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    Args:
        q: a 4-dim numpy array corresponding to a quaternion
        to: a string, either 'xyzw' or 'wxyz', determining
            which convention to convert to.
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")


def quat_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.
    >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> np.allclose(q, [-44, -14, 48, 28])
    True
    """
    x0, y0, z0, w0 = np.split(quaternion0, 4, axis=-1)  # (..., 1) for each
    x1, y1, z1, w1 = np.split(quaternion1, 4, axis=-1)
    return np.concatenate(
        [
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,  # (..., 1)
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ],
        axis=-1
    )


def quat_difference(quaternion1, quaternion2):
    return quat_multiply(quaternion1, quat_inverse(quaternion2))


def quat_conjugate(quaternion):
    """Return conjugate of quaternion.
    >>> q0 = random_quaternion()
    >>> q1 = quat_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True
    """
    quaternion = np.asarray(quaternion)
    return np.stack(
        [-quaternion[..., 0], -quaternion[..., 1], -quaternion[..., 2], quaternion[..., 3]],
        axis=-1
    )


def quat_inverse(quaternion):
    """Return inverse of quaternion.
    >>> q0 = random_quaternion()
    >>> q1 = quat_inverse(q0)
    >>> np.allclose(quat_multiply(q0, q1), [0, 0, 0, 1])
    True
    """
    quaternion = np.asarray(quaternion)
    return quat_conjugate(quaternion) / np.sum(quaternion * quaternion, axis=-1, keepdims=True)


def quat_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.
    >>> q0 = random_quat()
    >>> q1 = random_quat()
    >>> q = quat_slerp(q0, q1, 0.0)
    >>> np.allclose(q, q0)
    True
    >>> q = quat_slerp(q0, q1, 1.0, 1)
    >>> np.allclose(q, q1)
    True
    >>> q = quat_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2.0, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2.0, math.acos(-np.dot(q0, q1)) / angle)
    True
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])

    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    d = np.clip(d, -1.0, 1.0)
    angle = math.acos(d) + spin * math.pi

    if abs(angle) < EPS:
        return q0
    isin = 1.0 / math.sin(angle)

    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1

    return q0


def batch_quat_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.
    >>> q0 = random_quat()
    >>> q1 = random_quat()
    >>> q = quat_slerp(q0, q1, 0.0)
    >>> np.allclose(q, q0)
    True
    >>> q = quat_slerp(q0, q1, 1.0, 1)
    >>> np.allclose(q, q1)
    True
    >>> q = quat_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2.0, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2.0, math.acos(-np.dot(q0, q1)) / angle)
    True
    """
    fraction = np.broadcast_to(fraction, quat0.shape[:-1])
    fraction = fraction[..., None]

    q0 = quat0[..., :4]
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = quat1[..., :4]
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)

    assert len(q0.shape) == len(q1.shape) == 2, [q0.shape, q1.shape]

    ret = np.zeros_like(q0[..., :1], dtype=bool)
    ret_q = np.zeros_like(q0, dtype=bool)

    # base checks
    ret = ret | (fraction == 0.) | (fraction == 1.)
    ret_q = np.where(fraction == 0, q0, ret_q)
    ret_q = np.where(fraction == 1, q1, ret_q)

    # check for orthogonality?
    d = np.sum(q0 * q1, axis=-1, keepdims=True)  # dot
    small_d = np.abs(np.abs(d) - 1.0) < EPS
    ret_q = np.where(small_d & ~ret, q0, ret_q)
    ret = ret | small_d

    if shortestpath:
        # invert rotation
        q1 = np.where(d < 0.0, -q1, q1)
        d = np.where(d < 0.0, -d, d)

    d = np.clip(d, -1.0, 1.0)
    angle = np.arccos(d) + spin * np.pi
    small_ang = np.abs(angle) < EPS
    ret_q = np.where(small_ang & ~ret, q0, ret_q)
    ret = ret | small_ang

    remaining = ~ret[..., 0]
    # only over the remaining, to avoid divide by zero issues
    isin = 1.0 / np.sin(angle[remaining])
    q0_r = q0[remaining] * np.sin((1.0 - fraction[remaining]) * angle[remaining]) * isin
    q1_r = q1[remaining] * np.sin(fraction[remaining] * angle[remaining]) * isin
    q0_r += q1_r

    ret_q[remaining] = q0_r
    return ret_q


def random_quat(rand=None):
    """Return uniform random unit quaternion.
    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.
    >>> q = random_quat()
    >>> np.allclose(1.0, vector_norm(q))
    True
    >>> q = random_quat(np.random.random(3))
    >>> q.shape
    (4,)
    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array(
        (np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2, np.cos(t2) * r2),
        dtype=np.float32,
    )


# TODO: does T have extrinsic rotations?
euler2quat_intrinsic = lambda euler: T.euler_to_quat(euler)[[1, 2, 3, 0]]
quat2euler_intrinsic = lambda quat: T.quat_to_euler(quat[[3, 0, 1, 2]])

# extrinsic
euler2quat = lambda euler: R.from_euler("xyz", euler).as_quat()
quat2euler = lambda quat: R.from_quat(quat).as_euler("xyz")

# mat2quat = lambda mat: T.mat_to_quat(mat)[[1,2,3,0]]
# quat2mat = lambda quat: T.quat_to_mat(quat[[3,0,1,2]])[:3, :3]

mat2quat = lambda mat: R.from_matrix(mat).as_quat()
quat2mat = lambda quat: R.from_quat(quat).as_matrix()


def add_euler(delta, source, degrees=False):
    delta_rot = R.from_euler('xyz', delta, degrees=degrees)
    source_rot = R.from_euler('xyz', source, degrees=degrees)
    new_rot = delta_rot * source_rot
    return new_rot.as_euler('xyz', degrees=degrees)


def add_quats(delta: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Perform quaternion addition =>> delta * source."""
    return (R.from_quat(delta) * R.from_quat(source)).as_quat()


def euler2mat(euler):
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat

def axisangle2quat(vec):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (np.array): (ax,ay,az) axis-angle exponential coordinates
    Returns:
        np.array: (x,y,z,w) vec4 float angles
    """
    front_shape = list(vec.shape[:-1])
    vec = vec.reshape(-1, 3)
    # Grab angle
    angle = np.linalg.norm(vec, axis=-1, keepdims=True)

    q = np.zeros((vec.shape[0], 4))
    zero_cond = np.isclose(angle, 0.)

    # make sure that axis is a unit vector
    axis = np.divide(vec, angle, out=vec.copy(), where=~zero_cond)

    q[..., 3:] = np.cos(angle / 2.)
    q[..., :3] = axis * np.sin(angle / 2.)

    # handle zero-rotation case
    q = np.where(zero_cond, np.array([0., 0., 0., 1.]), q)

    return q.reshape(front_shape + [4])


def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.
    Args:
        quat (np.array): (x,y,z,w) vec4 float angles
    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    quat[..., 3] = np.where(quat[..., 3] > 1, 1., quat[..., 3])
    quat[..., 3] = np.where(quat[..., 3] < -1, -1., quat[..., 3])

    den = np.sqrt(1. - quat[..., 3] * quat[..., 3])
    zero_cond = np.isclose(den, 0.)

    scale = np.divide(1., den, out=np.zeros_like(den), where=~zero_cond)

    return (quat[..., :3] * 2. * np.arccos(quat[..., 3])) * scale

mat2axisangle = lambda mat: quat2axisangle(mat2quat(mat))
axisangle2mat = lambda axa: quat2mat(axisangle2quat(axa))

def quat_angle(q1, q0):
    scalar = 2 * (q0 * q1).sum(-1) ** 2
    return np.arccos(np.clip(scalar - 1, -1, 1))


mat2euler = lambda mat, format="XYZ": T.rmat_to_euler(mat, format)


@numba.jit(nopython=True, cache=True)
def pose2mat(pose):
    """
    Converts pose to homogeneous matrix.

    Args:
        pose: a (pos, orn) tuple where pos is vec3 float cartesian, and
            orn is vec4 float quaternion.

    Returns:
        4x4 homogeneous matrix
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.
    return homo_pose_mat


@numba.jit(nopython=True, cache=True)
def calc_twist(jacobian, dq):
    """Calculate the twist (ee velocity and angular velocity)
    from jacobian and joint velocity.
    """
    return np.dot(jacobian, dq)

def pose_in_A_to_pose_in_B(pose_A, pose_A_in_B):
    """
    Converts a homogenous matrix corresponding to a point C in frame A
    to a homogenous matrix corresponding to the same point C in frame B.

    Args:
        pose_A: numpy array of shape (4,4) corresponding to the pose of C in frame A
        pose_A_in_B: numpy array of shape (4,4) corresponding to the pose of A in frame B

    Returns:
        numpy array of shape (4,4) corresponding to the pose of C in frame B
    """

    # pose of A in B takes a point in A and transforms it to a point in C.

    # pose of C in B = pose of A in B * pose of C in A
    # take a point in C, transform it to A, then to B
    # T_B^C = T_A^C * T_B^A
    return pose_A_in_B.dot(pose_A)


def pose_inv(pose):
    """
    Computes the inverse of a homogenous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    Args:
        pose: numpy array of shape (4,4) for the pose to inverse

    Returns:
        numpy array of shape (4,4) for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense.
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    # R-1 to align the axis again.

    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv


def _skew_symmetric_translation(pos_A_in_B):
    """
    Helper function to get a skew symmetric translation matrix for converting quantities
    between frames.
    """
    return np.array(
        [
            0.,
            -pos_A_in_B[2],
            pos_A_in_B[1],
            pos_A_in_B[2],
            0.,
            -pos_A_in_B[0],
            -pos_A_in_B[1],
            pos_A_in_B[0],
            0.,
        ]
    ).reshape((3, 3))


def vel_in_A_to_vel_in_B(vel_A, ang_vel_A, pose_A_in_B):
    """
    Converts linear and angular velocity of a point in frame A to the equivalent in frame B.

    Args:
        vel_A: 3-dim iterable for linear velocity in A
        ang_vel_A: 3-dim iterable for angular velocity in A
        pose_A_in_B: numpy array of shape (4,4) corresponding to the pose of A in frame B

    Returns:
        vel_B, ang_vel_B: two numpy arrays of shape (3,) for the velocities in B
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    vel_B = rot_A_in_B.dot(vel_A) + skew_symm.dot(rot_A_in_B.dot(ang_vel_A))
    ang_vel_B = rot_A_in_B.dot(ang_vel_A)
    return vel_B, ang_vel_B


def force_in_A_to_force_in_B(force_A, torque_A, pose_A_in_B):
    """
    Converts linear and rotational force at a point in frame A to the equivalent in frame B.

    Args:
        force_A: 3-dim iterable for linear force in A
        torque_A: 3-dim iterable for rotational force (moment) in A
        pose_A_in_B: numpy array of shape (4,4) corresponding to the pose of A in frame B

    Returns:
        force_B, torque_B: two numpy arrays of shape (3,) for the forces in B
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    force_B = rot_A_in_B.T.dot(force_A)
    torque_B = -rot_A_in_B.T.dot(skew_symm.dot(force_A)) + rot_A_in_B.T.dot(torque_A)
    return force_B, torque_B


def rotation_matrix(angle, direction, point=None):
    """
    Returns matrix to rotate about axis defined by point and direction.

    Examples:

        >>> angle = (random.random() - 0.5) * (2*math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True
        >>> I = numpy.identity(4, numpy.float32)
        >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
        True
        >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
        ...                                                direc, point)))
        True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(
        ((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float32
    )
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float32, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def clip_translation(dpos, limit):
    """
    Limits a translation (delta position) to a specified limit

    Scales down the norm of the dpos to 'limit' if norm(dpos) > limit, else returns immediately

    :param dpos: n-dim Translation being clipped (e,g.: (x, y, z)) -- numpy array
    :param limit: Value to limit translation by -- magnitude (scalar, in same units as input)
    :return: Clipped translation (same dimension as inputs)
    """
    input_norm = np.linalg.norm(dpos)
    return dpos * limit / input_norm if input_norm > limit else dpos


def clip_rotation(quat, limit):
    """
    Limits a (delta) rotation to a specified limit

    Converts rotation to axis-angle, clips, then re-converts back into quaternion

    :param quat: Rotation being clipped (x, y, z, w) -- numpy array
    :param limit: Value to limit rotation by -- magnitude (scalar, in radians)
    :return: Clipped rotation quaternion (x, y, z, w)
    """
    # First, normalize the quaternion
    quat = quat / np.linalg.norm(quat)

    den = np.sqrt(max(1 - quat[3] * quat[3], 0))
    if den == 0:
        # This is a zero degree rotation, immediately return
        return quat
    else:
        # This is all other cases
        x = quat[0] / den
        y = quat[1] / den
        z = quat[2] / den
        a = 2 * math.acos(quat[3])

    # Clip rotation if necessary and return clipped quat
    if abs(a) > limit:
        a = limit * np.sign(a) / 2
        sa = math.sin(a)
        ca = math.cos(a)
        quat = np.array([
            x * sa,
            y * sa,
            z * sa,
            ca
        ])

    return quat

def make_pose(translation, rotation):
    """
    Makes a homogenous pose matrix from a translation vector and a rotation matrix.

    Args:
        translation: a 3-dim iterable
        rotation: a 3x3 matrix

    Returns:
        pose: a 4x4 homogenous matrix
    """
    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose


def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    Examples:

        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True
        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True
        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True
        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True
        >>> list(unit_vector([]))
        []
        >>> list(unit_vector([1.0]))
        [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def get_orientation_error(target_orn, current_orn):
    """
    Returns the difference between two quaternion orientations as a 3 DOF numpy array.
    For use in an impedance controller / task-space PD controller.

    Args:
        target_orn: 4-dim iterable, desired orientation as a (x, y, z, w) quaternion
        current_orn: 4-dim iterable, current orientation as a (x, y, z, w) quaternion

    Returns:
        orn_error: 3-dim numpy array for current orientation error, corresponds to
            (target_orn - current_orn)
    """
    current_orn = np.array(
        [current_orn[3], current_orn[0], current_orn[1], current_orn[2]]
    )
    target_orn = np.array([target_orn[3], target_orn[0], target_orn[1], target_orn[2]])

    pinv = np.zeros((3, 4))
    pinv[0, :] = [-current_orn[1], current_orn[0], -current_orn[3], current_orn[2]]
    pinv[1, :] = [-current_orn[2], current_orn[3], current_orn[0], -current_orn[1]]
    pinv[2, :] = [-current_orn[3], -current_orn[2], current_orn[1], current_orn[0]]
    orn_error = 2.0 * pinv.dot(np.array(target_orn))
    return orn_error


def get_pose_error(target_pose, current_pose):
    """
    Computes the error corresponding to target pose - current pose as a 6-dim vector.
    The first 3 components correspond to translational error while the last 3 components
    correspond to the rotational error.

    Args:
        target_pose: a 4x4 homogenous matrix for the target pose
        current_pose: a 4x4 homogenous matrix for the current pose

    Returns:
        A 6-dim numpy array for the pose error.
    """
    error = np.zeros(6)

    # compute translational error
    target_pos = target_pose[:3, 3]
    current_pos = current_pose[:3, 3]
    pos_err = target_pos - current_pos

    # compute rotational error
    r1 = current_pose[:3, 0]
    r2 = current_pose[:3, 1]
    r3 = current_pose[:3, 2]
    r1d = target_pose[:3, 0]
    r2d = target_pose[:3, 1]
    r3d = target_pose[:3, 2]
    rot_err = 0.5 * (np.cross(r1, r1d) + np.cross(r2, r2d) + np.cross(r3, r3d))

    error[:3] = pos_err
    error[3:] = rot_err
    return error

def convert_euler_quat_2mat(ori):
    """Convert an euler or quaternion to matrix for orientation error.
    """
    if len(ori) == 3:
        # Euler angles.
        return euler2mat(ori)
    elif len(ori) == 4:
        return quat2mat(ori)
    else:
        raise ValueError("Invalid orientation dim of len {}".format(len(ori)))


def fast_quat2euler(quat):
    # https://stackoverflow.com/questions/56207448/efficient-quaternions-to-euler-transformation
    assert quat.shape[-1] == 4
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)

    return np.stack([X, Y, Z], axis=-1)


def fast_euler2quat(eul):
    # https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr
    assert eul.shape[-1] == 3
    roll, pitch, yaw = eul[..., 0], eul[..., 1], eul[..., 2]

    sin_roll2 = np.sin(roll / 2)
    cos_roll2 = np.cos(roll / 2)

    sin_pitch2 = np.sin(pitch / 2)
    cos_pitch2 = np.cos(pitch / 2)

    sin_yaw2 = np.sin(yaw / 2)
    cos_yaw2 = np.cos(yaw / 2)

    sin_roll2_cos_pitch2 = sin_roll2 * cos_pitch2
    cos_roll2_sin_pitch2 = cos_roll2 * sin_pitch2

    cos_roll2_cos_pitch2 = cos_roll2 * cos_pitch2
    sin_roll2_sin_pitch2 = sin_roll2 * sin_pitch2

    qx = sin_roll2_cos_pitch2 * cos_yaw2 - cos_roll2_sin_pitch2 * sin_yaw2
    qy = cos_roll2_sin_pitch2 * cos_yaw2 + sin_roll2_cos_pitch2 * sin_yaw2
    qz = cos_roll2_cos_pitch2 * sin_yaw2 - sin_roll2_sin_pitch2 * cos_yaw2
    qw = cos_roll2_cos_pitch2 * cos_yaw2 + sin_roll2_sin_pitch2 * sin_yaw2

    return np.stack([qx, qy, qz, qw], axis=-1)


if __name__ == '__main__':
    from muse.utils.general_utils import timeit

    for i in range(100):
        euler = np.random.uniform([-np.pi, -np.pi, -np.pi/2], [np.pi, np.pi, np.pi/2])
        quat = euler2quat_intrinsic(euler)
        mat = quat2mat(quat)
        q_err = get_orientation_error(quat, mat2quat(mat))
        assert np.linalg.norm(q_err) < 1e-5, q_err

        # euler_again = mat2euler(mat)
        rand_quat = random_quat()
        r_euler = mat2euler(quat2mat(rand_quat))
        r_quat_again = euler2quat_intrinsic(r_euler)
        # r_quat_again = euler2quat(r_euler)
        # euler_again = dm_control.utils.transformations.rmat_to_euler(mat, 'XYZ')
        # other = euler2quat(euler_again)
        # other = dm_control.utils.transformations.euler_to_quat(euler_again)[[1, 2, 3, 0]]
        q_err = get_orientation_error(rand_quat, r_quat_again)
        assert np.linalg.norm(q_err) < 1e-5, q_err

    # extrinsic fast test
    for i in range(1000):
        eul = np.random.uniform([-np.pi, -np.pi / 2, -np.pi], [np.pi, np.pi / 2, np.pi], size=(3000, 3))
        with timeit("euler2quat"):
            quat = euler2quat(eul)
        with timeit("fast_euler2quat"):
            quat_fast = fast_euler2quat(eul)

        # testing euler -> quat
        q_err = quat_angle(quat, quat_fast)
        assert np.linalg.norm(q_err) < 1e-5, q_err

        # testing quat -> euler
        with timeit("quat2euler"):
            _ = quat2euler(quat)

        with timeit("fast_quat2euler"):
            eul_fast_from_quat = fast_quat2euler(quat)

        quat_of_eul_fast_from_quat = euler2quat(eul_fast_from_quat)
        q_err2 = quat_angle(quat, quat_of_eul_fast_from_quat)
        assert np.linalg.norm(q_err2) < 1e-5, q_err2

    print(timeit)
    timeit.reset()

    # QUAT multiply check.
    for i in range(1000):
        rand_quat = random_quat()
        delta_quat = random_quat()
        rand_quat /= np.linalg.norm(rand_quat)
        delta_quat /= np.linalg.norm(delta_quat)

        with timeit("quat_mul"):
            composed = quat_multiply(delta_quat, rand_quat)
        with timeit("rot_add_quats"):
            composed_real = add_quats(delta_quat, rand_quat)
        q_err2 = quat_angle(composed, composed_real)
        assert np.linalg.norm(q_err2) < 1e-3, [q_err2, composed, composed_real]

    print(timeit)
    timeit.reset()
    print("done.")


def convert_rpt(roll, phi, theta, bias=np.array([0, -np.pi/2, 0])):

    # returns quat and eul
    rot1 = R.from_rotvec([0, 0, bias[0] + roll])
    rot2 = R.from_rotvec([bias[1] + phi, 0, 0])
    rot3 = R.from_rotvec([0, 0, bias[2] + theta])
    chained = rot3 * rot2 * rot1
    return chained.as_quat(), chained.as_euler("xyz")


def convert_eul_to_rpt(eul, bias=np.array([0, -np.pi/2, 0])):
    # returns quat and eul
    chained = R.from_euler("xyz", eul)
    rpt = chained.as_euler("zxz") - bias
    return rpt


def convert_rpt_to_quat_eul(rpt, bias=np.array([0, -np.pi/2, 0])):
    rot = R.from_euler("zxz", rpt + bias)
    return rot.as_quat(), rot.as_euler("xyz")


def convert_quat_to_rpt(quat, bias=np.array([0, -np.pi/2, 0])):
    # returns r,p,t
    chained = R.from_quat(quat)
    rpt = chained.as_euler("zxz") - bias
    return rpt


circ = [False] * 3 + [True] * 3
def pose_difference_fn(pose1, pose2):
    return np.array(
        [circular_difference(pose1[i], pose2[i]) if circ[i] else pose1[i] - pose2[i] for i in range(len(pose1))])


def rot6d2mat(d6: np.ndarray) -> np.ndarray:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    Adapted from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_rotation_6d
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def mat2rot6d(matrix: np.ndarray) -> np.ndarray:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    Adapted from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_rotation_6d
    """
    batch_dim = list(matrix.shape[:-2])
    return matrix[..., :2, :].copy().reshape(batch_dim + [6])
