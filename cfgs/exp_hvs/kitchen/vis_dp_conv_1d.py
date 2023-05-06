from cfgs.env import kitchen
from cfgs.exp_hvs.square import vis_dp_conv_1d as sq_vis_dp_conv_1d
from cfgs.model import vis_bc_rnn
from cfgs.trainer import rm_goal_trainer

from attrdict import AttrDict as d

from configs.fields import Field as F

from muse.envs.robosuite.robosuite_env import RobosuiteEnv
from muse.envs.robosuite.robosuite_utils import modify_spec_prms, get_rs_online_action_postproc_fn
from muse.policies.memory_policy import get_timeout_terminate_fn

env_spec_prms = RobosuiteEnv.get_default_env_spec_params(kitchen.export)
env_spec_prms = modify_spec_prms(env_spec_prms, no_object=True)

export = sq_vis_dp_conv_1d.export.node_leaf_without_keys(['dataset', 'env_spec', 'env_train']) & d(
    augment=False,
    batch_size=16,
    dataset='human_buds-kitchen_60k_eimgs',
    exp_name='hvsBlock3D/velact_{?augment:aug_}b{batch_size}_h{horizon}_{dataset}',

    model=d(
        state_names=['robot0_eef_pos', 'robot0_gripper_qpos'],
        vision_encoder=d(
            image_shape=[128, 128, 3],
            img_embed_size=128,
        ),
    ),
    policy=d(
        online_action_postproc_fn=get_rs_online_action_postproc_fn(no_ori=True, fast_dynamics=True),
        is_terminated_fn=get_timeout_terminate_fn(1200),
    ),
    trainer=rm_goal_trainer.export & d(
        max_steps=1000000,
        train_do_data_augmentation=F('augment'),
    ),
)
