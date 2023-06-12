import numpy as np
from attrdict import AttrDict as d

from cfgs.exp_hvs.square import hydra
from configs.fields import GroupField
from muse.envs.robosuite.robosuite_env import RobosuiteEnv
from muse.envs.robosuite.robosuite_utils import get_wp_dynamics_fn, modify_spec_prms


def hydra_rot6_spec_mod(params):
    env_spec = modify_spec_prms(RobosuiteEnv.get_default_env_spec_params(params),
                                include_mode=True, include_target_names=True, target_rot6d=True)
    env_spec.names_shapes_limits_dtypes.append(("smooth_mode", (1,), (0, 255), np.uint8))
    env_spec.names_shapes_limits_dtypes.append(("mask", (1,), (0, 1), np.uint8))
    env_spec.wp_dynamics_fn = get_wp_dynamics_fn(fast_dynamics=True)
    return env_spec


export = hydra.export & d(
    dataset='mode_real2_v2_fast_human_square_30k_rot6d_imgs',
    env_spec=GroupField('env_train', hydra_rot6_spec_mod),
)
