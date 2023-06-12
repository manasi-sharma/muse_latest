import numpy as np
from attrdict import AttrDict as d

from cfgs.env import square
from cfgs.exp_hvs.square import dp_conv_1d
from muse.envs.robosuite.robosuite_env import RobosuiteEnv

export = dp_conv_1d.export & d(
    # this dataset has position actions relabeled, but using 6D rotations..
    dataset='human_square_abs6d_30k',
    exp_name='hvsBlock3D/posact_{?seed:s{seed}_}b{batch_size}_h{horizon}{?use_ema:_ema}_{dataset}',
    # change in the env
    env_spec=RobosuiteEnv.get_default_env_spec_params(square.export & d(use_delta=False, use_rot6d=True)),
    env_train=square.export & d(use_delta=False, use_rot6d=True),
    # NO ACTION NORMALIZING
    model=d(
        action_decoder=d(
            use_tanh_out=False,  # actions are not -1 to 1
        ),
    ),
)
