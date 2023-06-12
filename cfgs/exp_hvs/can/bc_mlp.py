from attrdict import AttrDict as d

from cfgs.env import can
from cfgs.exp_hvs.square import bc_mlp

from configs.fields import GroupField
from muse.envs.robosuite.robosuite_env import RobosuiteEnv

export = bc_mlp.export & d(
    dataset='human_can_25k',
    env_spec=GroupField('env_train', RobosuiteEnv.get_default_env_spec_params),
    env_train=can.export,
)
