from attrdict import AttrDict as d

from cfgs.exp_hvs.pusht import bc_rnn as pt_bc_rnn
from cfgs.model import bc_mlp

from configs.fields import Field as F

export = pt_bc_rnn.export.node_leaf_without_keys(['model']) & d(
    # MLP model
    model=bc_mlp.export & d(
        goal_names=[],
        state_names=F('use_keypoint', lambda kp: ['state', 'keypoint'] if kp else ['state']),
        device=F('device'),
        action_decoder=d(
            hidden_size=400,
            use_tanh_out=False,
        )
    ),
)
