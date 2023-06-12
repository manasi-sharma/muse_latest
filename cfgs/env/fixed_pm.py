import numpy as np
from attrdict import AttrDict as d

from muse.envs.simple.point_mass_env import PointMassEnv

export = PointMassEnv.default_params & d(
    cls=PointMassEnv,
    num_steps=50,
    initial_obs=np.array([0.1, 0.1]),
    initial_target=np.array([0.9, 0.9]),
)
