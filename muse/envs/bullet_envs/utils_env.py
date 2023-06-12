import os

import numpy as np
import pybullet as p
import torch
from enum import Enum

from muse.envs.bullet_envs.control_utils import orientation_error, batch_orientation_eul_add
from muse.utils.np_utils import clip_norm
from muse.utils.torch_utils import to_numpy
from muse.utils.transform_utils import euler2mat


def raw_get_view(width=600, height=600, look=[-0.05, -0.3, 0.0], dist=0.25, direct=[0.0, 0.0, 0.0]):
    cameraRandom = 0.0
    pitch = direct[0] + cameraRandom * np.random.uniform(-3, 3)
    yaw = direct[1] + cameraRandom * np.random.uniform(-3, 3)
    roll = direct[2] + cameraRandom * np.random.uniform(-3, 3)
    viewmatrix = p.computeViewMatrixFromYawPitchRoll(look, dist, yaw, pitch, roll, 2)

    fov = 40. + cameraRandom * np.random.uniform(-2, 2)
    aspect = float(width) / float(height)
    near = 0.01
    far = 10
    projmatrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    return viewmatrix, projmatrix


def get_view(opt):
    width = 640
    height = 480
    # TODO: Use config
    params_file = os.path.join(os.path.dirname(__file__), "camera_views/params.npy")
    params = np.load(params_file)
    if opt.view_point == 'third':
        dist = params[5] + 0.3
        look = [params[3] - 0.4, -params[4], 0.0]
        direct = [params[0] + 90, params[2] + 180, params[1]]
    else:
        dist = params[5]
        look = [params[3], params[4], 0.0]
        direct = [params[0]+90,params[2],params[1]]
    view_matrix,proj_matrix = raw_get_view(width,height,look,dist,direct)
    return view_matrix,proj_matrix

def get_view_sim():
    def getview (width=600, height=600, look=[-0.05, -0.3, 0.0], dist=0.25, direct=[0.0, 0.0, 0.0]):
        cameraRandom = 0.0
        pitch = direct[0] + cameraRandom * np.random.uniform (-3, 3)
        yaw = direct[1] + cameraRandom * np.random.uniform (-3, 3)
        roll = direct[2] + cameraRandom * np.random.uniform (-3, 3)
        viewmatrix = p.computeViewMatrixFromYawPitchRoll (look, dist, yaw, pitch, roll, 2)

        fov = 40. + cameraRandom * np.random.uniform (-2, 2)
        aspect = float (width) / float (height)
        near = 0.01
        far = 10
        projmatrix = p.computeProjectionMatrixFOV (fov, aspect, near, far)

        return viewmatrix, projmatrix

    width = 640
    height = 480
    params = np.load('../../configs/camera_parameters/params.npy')
    # dist = params[5]
    dist = params[5]+0.3
    look = [params[3]-0.4, -params[4], 0.0]
    direct = [params[0]+90,params[2]+180,params[1]]
    view_matrix,proj_matrix = getview(width,height,look,dist,direct)
    return view_matrix,proj_matrix

def safe_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def point2traj(points=None,delta=0.01):
    traj = []
    last = points[0]
    for i in range(len(points)-1):
        now = points[i+1]
        diff = [x-y for x,y in zip(now,last)]
        dist = sum([x**2 for x in diff])**0.5
        n = int(dist/delta)
        for step in range(n):
            x = last[0] + step*delta*diff[0]/dist
            y = last[1] + step*delta*diff[1]/dist
            z = last[2] + step*delta*diff[2]/dist
            traj.append([x,y,z])
        last = now
    return traj

def get_gripper_pos(gripperOpen=1):
    """
    :param gripperOpen: 1 represents open, 0 represents close
    :return: the gripperPos
    """

    gripperLowerLimitList = [0] * 6
    gripperUpperLimitList = [0.81, -0.8, 0.81, -0.8, 0.8757, 0.8757]

    gripperPos = np.array (gripperUpperLimitList) * (1 - gripperOpen) + np.array (gripperLowerLimitList) * gripperOpen
    return gripperPos

def cut_frame(video_name,output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    cmd = 'ffmpeg -i {} -vf scale=640:480 '.format(video_name)+'{}/%06d.jpg'.format(output_path)
    os.system(cmd)
    return output_path


class RobotControllerMode(Enum):
    x_pid = 0  # op space positional control
    xdot_pid = 1  # op space positional control
    xddot_pid = 2  # op space acceleration (torque + posture) control PID
    xddot_with_force_pid = 3  # op space acceleration (torque + posture) with op space forces, x, xdot
    xddot_with_zero_force_pid = 4  # op space acceleration (torque + posture) with compliant wrist forces, x, xdot
    direct_xddot_with_force_pid = 5  # op space acceleration with op space forces, xddot specified
    q_pid = 6  # joint space position control
    qdot_pid = 7  # joint space velocity control


def target_action_postproc_fn(obs, act, Kv_P=(30., 30., 10.), Kv_O=(30., 30., 30.), Kv_G=20.,
                                 max_vel=0.4, max_orn_vel=10.0, max_gripper_vel=150, relative=False, dt=0.1,
                                 clip_targ_orn=False, use_gripper=True, action_name="action", gripper_pos_name="gripper_pos", **kwargs):
    pos = to_numpy(obs >> "ee_position", check=True)
    ori_eul = to_numpy(obs >> "ee_orientation_eul", check=True)
    targ_pos = to_numpy(act >> "target/ee_position", check=True)
    targ_orn = to_numpy(act >> "target/ee_orientation_eul", check=True)
    if relative:
        raise NotImplementedError
    Kv_P = np.asarray(Kv_P) if isinstance(targ_pos, np.ndarray) else torch.tensor(Kv_P, device=targ_pos.device)
    Kv_O = np.asarray(Kv_O) if isinstance(targ_orn, np.ndarray) else torch.tensor(Kv_O, device=targ_orn.device)
    # yaw only if not in drawer env

    # targ_orn[..., 2] = np.where(targ_orn[..., 2] < -np.pi/2, targ_orn[..., 2] + np.pi, targ_orn[..., 2])
    ac_vel = Kv_P * (targ_pos - pos)
    ac_orn_vel = Kv_O * orientation_error(euler2mat(targ_orn.reshape(3)),
                                          euler2mat(ori_eul.reshape(3)))
    # B x H x N,   max over N blocks
    ac_dx = clip_norm(ac_vel, max_vel) * dt
    ac_orn_dx = clip_norm(ac_orn_vel, max_orn_vel) * dt

    if use_gripper:
        gripper_pos = to_numpy(obs >> gripper_pos_name, check=True)
        targ_grip = to_numpy(act >> f"target/{gripper_pos_name}", check=True)
        Kv_G = np.asarray(Kv_G) if isinstance(targ_grip, np.ndarray) else torch.tensor(Kv_G, device=targ_grip.device)
        ac_grab = Kv_G * (targ_grip - gripper_pos)
    else:
        gripper_pos = np.array([[[0]]])
        ac_grab = np.array([[[0]]])

    ac_grab_dx = clip_norm(ac_grab, max_gripper_vel) * dt

    clip_next_gripper = np.clip(gripper_pos + ac_grab_dx, 0, 250)  # max gripper is 250, not 255

    # print(obs.ee_orientation_eul, ac_orn_vel, batch_orientation_eul_add(obs.ee_orientation_eul, ac_orn_vel * DT))
    # print(ac_grab.shape, ac_vel.shape, norm_out.leaf_apply(lambda arr: arr.shape))
    ac = np.concatenate([pos + ac_dx, batch_orientation_eul_add(ac_orn_dx.reshape(ori_eul.shape), ori_eul), clip_next_gripper],
                        axis=-1)

    act = act.leaf_apply(lambda arr: arr[:, 0])
    act[action_name] = ac[:, 0]

    return act
