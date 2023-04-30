from attrdict import AttrDict as d

from cfgs.env import square
from cfgs.exp_hvs.square import bc_rnn
from cfgs.model import bc_mlp

from configs.fields import Field as F, GroupField
from muse.envs.robosuite.robosuite_env import RobosuiteEnv

export = bc_rnn.export.node_leaf_without_keys(['model']) & d(

    env_spec=GroupField('env_train', RobosuiteEnv.get_default_env_spec_params),
    env_train=square.export,

    # True = use of fixed scale norms (1 for pos, 10 for ori, and 100 for gripper)
    model=bc_mlp.export & d(
        device=F('device'),
        action_decoder=d(
            mlp_size=200,
        )
    ),

    policy=d(
        recurrent=False,
    ),
)