import numpy as np
from attrdict.utils import get_with_default

from muse.envs.simple.point_mass_env import PointMassEnv
from muse.policies.policy import Policy
from muse.utils.torch_utils import to_numpy

from attrdict import AttrDict as d

from muse.utils.transform_utils import unit_vector


class PMDirectPolicy(Policy):

    def _init_params_to_attrs(self, params):
        # max speed by default
        self.speed = get_with_default(params, "speed", 1.0)
        self.noise_std = get_with_default(params, "noise_std", 0.)

    def _init_setup(self):
        assert isinstance(self._env, PointMassEnv) and self._env.num_obstacles == 0

    def warm_start(self, model, observation, goal):
        pass

    def get_action(self, model, observation, goal, **kwargs):
        """
        :param model: (Model)
        :param observation: (AttrDict)  (B x H x ...)
        :param goal: (AttrDict) (B x H x ...)

        :return action: AttrDict (B x ...)
        """
        ego = to_numpy(observation.ego, check=True).reshape(2)
        targ = to_numpy(observation.target, check=True).reshape(2)

        dist = np.linalg.norm(targ - ego)

        # clip the distance traveled
        if self._env.ego_speed * self.speed > dist:
            # move the exact distance
            vel = (targ - ego) / self._env.ego_speed
        else:
            # vector pointing to target, at self.speed
            vel = (targ - ego) / dist * self.speed

        # noise_std scaled to match environment noise_std
        if self.noise_std > 0.:
            vel[:] += np.random.normal(0, self.noise_std, 2) / self._env.ego_speed

        return d(
            action=vel[None],
        )


class PMObstacleDirectPolicy(Policy):

    def _init_params_to_attrs(self, params):
        # 2D gaussian noise
        self.noise_std = get_with_default(params, "noise_std", 0.)
        # 1D angular gaussian noise (degrees)
        self.theta_noise_std = get_with_default(params, "theta_noise_std", 0.)
        self.random_side = get_with_default(params, "random_side", True)

        # max speed by default
        self.speed = get_with_default(params, "speed", 1.0)
        # addition to desired speed, will be clipped to [0., 1.]
        self.speed_noise_std = get_with_default(params, "speed_noise_std", 0.)
        # likelihood of stopping
        self.stop_prob = get_with_default(params, "stop_prob", 0.)
        # how long to stop for (uniformly sampled)
        self.stop_duration_range = get_with_default(params, "stop_duration", [2, 8])

    def _init_setup(self):
        assert isinstance(self._env, PointMassEnv) and self._env.num_obstacles == 1
        self.obstacle_loc = None
        self.mid_point = None
        self.reached_midpoint = False

        self.stop_steps = 0

    def warm_start(self, model, observation, goal):
        pass

    def reset_policy(self, next_obs=None, next_goal=None, **kwargs):
        self.obstacle_loc = next_obs.objects.position.reshape(2)
        self.obstacle_radii = next_obs.objects.size.item()
        top = np.random.random() > 0.5 if self.random_side else True
        line = self.obstacle_loc - next_obs.ego.reshape(2)
        if top:
            rot90_line = unit_vector(np.array([-line[1], line[0]]))
            t = min((1 - self.obstacle_loc[1]) / rot90_line[1], - self.obstacle_loc[0] / rot90_line[0])
        else:
            rot90_line = unit_vector(np.array([line[1], -line[0]]))
            t = min((1 - self.obstacle_loc[0]) / rot90_line[0], - self.obstacle_loc[1] / rot90_line[1])
        self.mid_point = rot90_line * t / 2 + self.obstacle_loc
        self.reached_midpoint = False

    def update_velocity(self, vel):
        vel = vel.copy()

        # theta noise std
        if self.theta_noise_std > 0.:
            theta = np.arctan2(vel[1], vel[0])
            norm = np.linalg.norm(vel)
            theta += np.random.normal(0, np.deg2rad(self.theta_noise_std))
            vel[:] = norm * np.array([np.cos(theta), np.sin(theta)])

        if self.speed_noise_std > 0.:
            norm = np.linalg.norm(vel)
            # slow down only
            new_norm = np.clip(norm - np.abs(np.random.normal(0, self.speed_noise_std)), 0, 1.)
            vel[:] = new_norm / (norm + 1e-11) * vel

        # noise_std scaled to match environment noise_std
        if self.noise_std > 0.:
            vel[:] += np.random.normal(0, self.noise_std, 2) / self._env.ego_speed

        if self.stop_steps == 0 and np.random.rand() < self.stop_prob:
            self.stop_steps = np.random.randint(*self.stop_duration_range)

        if self.stop_steps > 0:
            vel[:] = 0.
            self.stop_steps -= 1

        return vel

    def get_action(self, model, observation, goal, **kwargs):
        """
        :param model: (Model)
        :param observation: (AttrDict)  (B x H x ...)
        :param goal: (AttrDict) (B x H x ...)

        :return action: AttrDict (B x ...)
        """
        ego = to_numpy(observation.ego, check=True).reshape(2)

        if self.reached_midpoint:
            targ = to_numpy(observation.target, check=True).reshape(2)
        else:
            targ = self.mid_point

        dist = np.linalg.norm(targ - ego)

        # clip the distance traveled
        if self._env.ego_speed * self.speed > dist:
            # move the exact distance
            vel = (targ - ego) / self._env.ego_speed
        else:
            # vector pointing to target, at self.speed
            vel = (targ - ego) / dist * self.speed

        if not self.reached_midpoint and np.linalg.norm(targ - (ego + vel * self._env.ego_speed)) < 0.01:
            self.reached_midpoint = True

        vel = self.update_velocity(vel)

        return d(
            action=vel[None],
        )
