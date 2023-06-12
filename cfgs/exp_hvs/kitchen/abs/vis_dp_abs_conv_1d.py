from attrdict import AttrDict as d

from cfgs.env import kitchen
from cfgs.exp_hvs.kitchen import vis_dp_conv_1d

export = vis_dp_conv_1d.export & d(
    # this dataset has position actions relabeled..
    dataset='human_buds-kitchen_abs_60k_eimgs',
    # change in the env
    env_train=kitchen.export & d(use_delta=False),
    # stuff for normalization of actions, during training

    model=d(
        do_minmax_norm=True,
        normalize_actions=True,
        save_action_normalization=True,
        action_decoder=d(
            use_tanh_out=False,  # actions are not -1 to 1
        ),
    ),
    # policy also needs to normalize actions, during inference
    policy=d(
        policy_out_norm_names=['action']
    ),
)
