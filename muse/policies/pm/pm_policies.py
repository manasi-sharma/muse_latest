import numpy as np
from attrdict.utils import get_with_default

from muse.envs.simple.point_mass_env import PointMassEnv
from muse.policies.policy import Policy
from muse.utils.torch_utils import to_numpy

from attrdict import AttrDict as d


class PMDirectPolicy(Policy):

    def _init_params_to_attrs(self, params):
        # max speed by default
        self.speed = get_with_default(params, "speed", 1.0)
        self.variance = get_with_default(params, "variance", 0.)

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
            vel = (targ - ego) * self._env.ego_speed
        else:
            # vector pointing to target, at self.speed
            vel = (targ - ego) / dist * self.speed

        if self.variance > 0.:
            vel[:] += np.random.normal(0, self.variance, 2)

        return d(
            action=vel[None],
        )
