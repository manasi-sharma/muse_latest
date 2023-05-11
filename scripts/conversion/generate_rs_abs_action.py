import shutil
import sys

import numpy as np
import copy

import h5py
from attrdict import AttrDict as d
from scipy.spatial.transform import Rotation

from configs.helpers import load_base_config, get_script_parser
from muse.envs.robosuite.robosuite_env import RobosuiteEnv
from muse.experiments import logger


class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path, params, demo_prefix='demo_'):
        env_spec = params.env_spec.cls(params.env_spec)

        env = params.env_train.cls(params.env_train & d(use_delta=True), env_spec)
        assert isinstance(env, RobosuiteEnv)
        assert len(env.rs_env.robots) in (1, 2)

        abs_env = params.env_train.cls(params.env_train & d(use_delta=False), env_spec)
        assert not abs_env.rs_env.robots[0].controller.use_delta

        self.env = env
        self.abs_env = abs_env
        self.demo_prefix = demo_prefix
        self.file = h5py.File(dataset_path, 'r')
        logger.debug('Base keys:', self.file['data'].keys())

    def __len__(self):
        return len(list(k for k in self.file['data'].keys() if k.startswith(self.demo_prefix)))

    def convert_actions(self,
                        states: np.ndarray,
                        actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """

        env = self.env
        no_ori = env._no_ori
        ORI_DIM = 0 if no_ori else 3

        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        stacked_actions = actions.reshape(*actions.shape[:-1], -1, 4+ORI_DIM)

        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1] + (3,),
            dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1] + (ORI_DIM,),
            dtype=stacked_actions.dtype)
        action_gripper = stacked_actions[..., [-1]]

        self.env_reset_to(env, {'states': states[0]}, initial=True)
        # reset hard to the first state
        for i in range(len(states)):
            self.env_reset_to(env, {'states': states[i]})

            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.rs_env.robots):
                # run controller goal generator
                robot.control(stacked_actions[i, idx], policy_step=True)

                # read pos and ori from robots
                controller = robot.controller
                action_goal_pos[i, idx] = controller.goal_pos
                if not no_ori:
                    action_goal_ori[i, idx] = Rotation.from_matrix(
                        controller.goal_ori).as_rotvec()

        stacked_abs_actions = np.concatenate([
            action_goal_pos,
            action_goal_ori,
            action_gripper
        ], axis=-1)
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def env_reset_to(self, env, state, initial=False):
        env.reset_to(state)

        if initial:
            # some initial
            env.rs_env.sim.set_state_from_flattened(state['states'])

            # # pre-steps
            for _ in range(5):
                env.rs_env.step(env.zero_action)

    def convert_idx(self, idx, state_key="states"):
        file = self.file
        demo = file[f'data/{self.demo_prefix}{idx}']
        # input
        actions = demo['actions'][:]
        states = demo[state_key][-len(actions):]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)
        return abs_actions

    def convert_and_eval_idx(self, idx, state_key="states"):
        env = self.env
        abs_env = self.abs_env
        file = self.file
        # first step have high error for some reason, not representative
        eval_skip_steps = 1

        demo = file[f'data/{self.demo_prefix}{idx}']
        # input
        actions = demo['actions'][:]
        states = demo[state_key][-len(actions):]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)

        # verify
        # robot0_eef_pos = demo['obs']['robot0_eef_pos'][:]
        # robot0_eef_quat = demo['obs']['robot0_eef_quat'][:]
        robot0_eef_pos = demo['proprio_states'][:, :3]

        robot0_eef_quat = demo['proprio_states'][:, 3:7]

        delta_error_info = self.evaluate_rollout_error(
            env, states, actions, robot0_eef_pos, robot0_eef_quat,
            metric_skip_steps=eval_skip_steps)
        abs_error_info = self.evaluate_rollout_error(
            abs_env, states, abs_actions, robot0_eef_pos, robot0_eef_quat,
            metric_skip_steps=eval_skip_steps)

        info = {
            'delta_max_error': delta_error_info,
            'abs_max_error': abs_error_info
        }
        return abs_actions, info

    def evaluate_rollout_error(self, env,
                               states, actions,
                               robot0_eef_pos,
                               robot0_eef_quat,
                               metric_skip_steps=1):
        # first step have high error for some reason, not representative

        # evaluate abs actions
        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()
        obs = self.env_reset_to(env, {'states': states[0]}, initial=True)
        for i in range(len(states)):
            obs = self.env_reset_to(env, {'states': states[i]})
            obs, goal, done = env.step(d(action=actions[None, i]))
            rollout_next_states.append(env.get_state()['states'])
            rollout_next_eef_pos.append(obs['robot0_eef_pos'][0])
            rollout_next_eef_quat.append(obs['robot0_eef_quat'][0])
        rollout_next_states = np.array(rollout_next_states)
        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)

        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        print(next_eef_pos_dist[metric_skip_steps:metric_skip_steps+10])

        info = {
            'state': max_next_state_diff,
            'pos': max_next_eef_pos_dist,
        }
        if not env._no_ori:
            next_eef_rot_diff = Rotation.from_quat(robot0_eef_quat[1:]) \
                                * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
            next_eef_rot_dist = next_eef_rot_diff.magnitude()
            max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()
            info['rot'] = max_next_eef_rot_dist

        return info


if __name__ == '__main__':
    parser = get_script_parser()
    parser.add_argument('file', type=str, help="file to load for states to replay")
    parser.add_argument('out_file', type=str, help="file to load and add a key to")
    parser.add_argument('new_out_file', type=str, help="file to save the key to")
    parser.add_argument('--eval', action='store_true', help="if true, will eval and print out error")
    parser.add_argument('--demo_prefix', type=str, default='demo_')
    parser.add_argument('--state_key', type=str, default='states')
    parser.add_argument('config', type=str, help="common params for all modules.")
    local_args, unknown = parser.parse_known_args()

    logger.debug(f"Raw command: \n{' '.join(sys.argv)}")

    # load the config
    params, root = load_base_config(local_args.config, unknown)
    exp_name = root.get_exp_name()

    # instantiate classes from the params
    converter = RobomimicAbsoluteActionConverter(local_args.file, params, demo_prefix=local_args.demo_prefix)

    out_file = dict(np.load(local_args.out_file, allow_pickle=True))

    all_abs_actions = []

    for ep_idx in range(len(converter)):
        if local_args.eval:
            abs_actions, info = converter.convert_and_eval_idx(ep_idx, state_key=local_args.state_key)
            print(info['delta_max_error'], info['abs_max_error'])
        else:
            abs_actions = converter.convert_idx(ep_idx, state_key=local_args.state_key)
        logger.debug("Episode", ep_idx, "done.")

        all_abs_actions.append(abs_actions)

    abs_actions = np.concatenate(all_abs_actions)

    logger.debug('Before:', out_file['action'].shape)
    logger.debug('After:', abs_actions.shape)

    out_file['action'] = abs_actions

    logger.info('Saving...')
    np.savez_compressed(local_args.new_out_file, **out_file)

