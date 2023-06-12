import numpy as np

from muse.envs.env import make
from muse.envs.pymunk import teleop_functions
from muse.envs.pymunk.block_env_2d import BlockEnv2D
from muse.envs.pymunk.teleop_functions import pygame_key_teleop_step
from muse.experiments import logger
from attrdict import AttrDict as d
from attrdict.utils import get_with_default


class NavigationBlockEnv2D(BlockEnv2D):
    """
    Maze navigation environments.
    """
    def _init_params_to_attrs(self, params):
        super(NavigationBlockEnv2D, self)._init_params_to_attrs(params)
        self._tolerance = get_with_default(params, "tolerance", 10)
        self._use_dense_reward = get_with_default(params, "use_dense_reward", False)
        self._reward = 0

        assert self.valid_goal_idxs is not None, "Goal distribution must be provided for navigation env"

        self.posact_to_vel_fn = teleop_functions.get_posact_to_velact_fn(Kv_P=1. / self.dt, MAX_VEL=150.)
        self._teleop_fn = get_with_default(params, "teleop_fn",
                                           teleop_functions.get_pygame_mouse_teleop_fn(
                                               posact_to_vel_fn=self.posact_to_vel_fn))

    def _get_obs(self):
        obs = super(NavigationBlockEnv2D, self)._get_obs()
        dist = np.linalg.norm(self.goal_body.position - self.player_body.position)
        if self._use_dense_reward:
            self._reward = -dist
        else:
            self._reward = float(dist < self._tolerance)
        obs['reward'] = np.array([self._reward])[None]
        return obs

    def is_success(self) -> bool:
        return np.linalg.norm(self.goal_body.position - self.player_body.position) < self._tolerance


class BottleneckNavigationBlockEnv3D(NavigationBlockEnv2D):
    def _init_params_to_attrs(self, params):
        self.num_maze_cells = get_with_default(params, "num_maze_cells", 8, map_fn=int)

        # fixed maze
        self.bottleneck_indices = get_with_default(params, "bottleneck_indices", np.array([2, 6, 4]), map_fn=np.asarray)
        self.num_bn = len(self.bottleneck_indices)
        self.bn_width = self.num_maze_cells

        # random maze
        self.randomize_maze = get_with_default(params, "randomize_maze", False)
        self.skip_bottleneck_fn = get_with_default(params, "skip_bottleneck_fn", lambda *args: False)
        self.curr_bottleneck_indices = self.bottleneck_indices.copy()
        self.bottleneck_indices_bounds = get_with_default(params, "bottleneck_indices_bounds",
                                                          (np.array([0, 0, 0]),
                                                           np.array([self.bn_width, self.bn_width, self.bn_width])))

        # direction of maze lines.
        self.horizontal = get_with_default(params, "horizontal", True)

        if not self.randomize_maze:
            # fixed maze
            params.fixed_np_maze = BottleneckNavigationBlockEnv3D.generate_bottleneck_maze(self.bottleneck_indices,
                                                                                           self.horizontal,
                                                                                           self.num_maze_cells)
        else:
            logger.info("Randomizing maze on reset!")

        super(BottleneckNavigationBlockEnv3D, self)._init_params_to_attrs(params)

    def reset(self, presets: d = d()):
        if self.randomize_maze:
            presets = presets.leaf_copy()
            curr_bottleneck_indices = np.random.randint(*self.bottleneck_indices_bounds)
            while self.skip_bottleneck_fn(curr_bottleneck_indices):
                curr_bottleneck_indices = np.random.randint(*self.bottleneck_indices_bounds)

            self.curr_bottleneck_indices = get_with_default(presets, "maze_low_dim", curr_bottleneck_indices,
                                                            map_fn=lambda arr: arr[:self.num_bn])
            # logger.debug(self.curr_bottleneck_indices)
            maze = BottleneckNavigationBlockEnv3D.generate_bottleneck_maze(self.curr_bottleneck_indices,
                                                                           self.horizontal,
                                                                           self.num_maze_cells)
            presets.maze = get_with_default(presets, "maze", maze)

        bw = self.grid_size // (self.num_bn + 1)
        bw = np.flip(bw) if self.horizontal else bw
        cw = np.flip(self.cell_width) if self.horizontal else self.cell_width
        self.bn_pos_axis0 = bw[0] * (np.arange(self.num_bn) + 1)  # left to right
        if self.horizontal:
            self.bn_pos_axis0 = np.flip(self.bn_pos_axis0)  # top to bottom
        self.bn_posL_axis1 = cw[1] * self.curr_bottleneck_indices
        self.bn_posR_axis1 = cw[1] * (self.curr_bottleneck_indices + 1)
        self.bn_posC_axis1 = 0.5 * (self.bn_posL_axis1 + self.bn_posR_axis1)

        return super(BottleneckNavigationBlockEnv3D, self).reset(presets)

    def get_bottleneck_center_pos(self, i=None):
        all_center_pos = np.stack([self.bn_pos_axis0, self.bn_posC_axis1], axis=-1)  # N x 2
        if self.horizontal:
            all_center_pos = np.flip(all_center_pos, axis=1)
        return all_center_pos if i is None else all_center_pos[i]

    def _get_obs(self):
        obs = super(BottleneckNavigationBlockEnv3D, self)._get_obs()
        # low dimensional maze representation
        obs['maze_low_dim'] = np.concatenate([self.curr_bottleneck_indices, [int(self.horizontal)]])[None]
        return obs

    @staticmethod
    def generate_bottleneck_maze(missing_idxs, flipped, num_cells):
        num_passes = len(missing_idxs)
        spacing = num_cells // (num_passes + 1)  # one pass per spacing grid indices
        assert spacing > 0, "Num cells should be greater than num passes"
        assert num_cells % (num_passes + 1) == 0

        missing_idxs = np.array(missing_idxs)
        if flipped:
            L, R = 8, 2
        else:
            L, R = 1, 4

        if spacing <= 1:
            maze_row = [L] + [R + L] * (num_passes - 1) + [R]
            left_indices = list(range(num_passes))
        else:
            maze_row = [0]
            left_indices = []
            for i in range(num_passes):
                maze_row.extend([0] * max(spacing - 2, 0))
                maze_row.extend([L, R])
                left_indices.append(len(maze_row) - 2)
            maze_row.extend([0] * max(spacing - 1, 0))

        maze = np.array([maze_row for _ in range(num_cells)])
        for i in range(num_passes):
            maze[missing_idxs[i], left_indices[i]] -= L
            maze[missing_idxs[i], left_indices[i] + 1] -= R

        return np.flipud(maze.T) if flipped else maze

    default_params = BlockEnv2D.default_params & d(
        num_blocks=0,
        bottleneck_indices=(1, 3, 2),
        horizontal=True,
        ego_block_size=40,
        initialization_steps=5,
        num_maze_cells=8,
        do_wall_collisions=True,
        keep_in_bounds=True,
        valid_start_idxs=np.array([[0, 8 - 1]]),
        valid_goal_idxs=np.array([[8 - 1, 0]])
    )

    @staticmethod
    def get_default_env_spec_params(params=None):
        env_spec_params = BlockEnv2D.get_default_env_spec_params(params)
        env_spec_params["observation_names"].append('goal_position')
        return env_spec_params


if __name__ == "__main__":
    env_params = d(
        render=True,
        realtime=True,
    )

    env = make(BottleneckNavigationBlockEnv3D, env_params)

    # cv2.namedWindow("image_test", cv2.WINDOW_AUTOSIZE)

    env.reset()

    running = True
    gamma = 0.5
    last_act = np.zeros(3)
    act = np.zeros(3)
    while running:
        obs, goal, act, last_act, done, running = pygame_key_teleop_step(env, act, last_act, gamma)

        # print("Reward:", obs['reward'])

        # cv2.imshow("image_test", block.get_img())
        # cv2.waitKey(1)
