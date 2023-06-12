from attrdict import AttrDict as d

from cfgs.env import fixed_obstacle_pm
from cfgs.exp_hvs.pm import bc_mlp as pm_bc_mlp

from configs.fields import Field as F

export = pm_bc_mlp.export & d(
    dataset='all_pmobs_direct_1000ep',
    env_train=fixed_obstacle_pm.export,

    # True = use of fixed scale norms (1 for pos, 10 for ori, and 100 for gripper)
    model=d(
        state_names=['ego', 'target', 'objects/position', 'objects/size'],
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=d(
        batch_names_to_get=['ego', 'target', 'objects/position', 'objects/size', 'action'],
    ),
    dataset_holdout=d(
        initial_load_episodes=F('load_eps'),
        batch_names_to_get=['ego', 'target', 'objects/position', 'objects/size', 'action'],
    ),
)
