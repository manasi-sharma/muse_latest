from cfgs.dataset import np_img_base_seq
from cfgs.env import kitchen
from cfgs.exp_hvs.square import vis_dp_conv_1d as sq_vis_dp_conv_1d

from attrdict import AttrDict as d

from configs.fields import Field as F

from muse.envs.robosuite.robosuite_env import RobosuiteEnv
from muse.envs.robosuite.robosuite_utils import modify_spec_prms, get_rs_online_action_postproc_fn
from muse.policies.memory_policy import get_timeout_terminate_fn

env_spec_prms = RobosuiteEnv.get_default_env_spec_params(kitchen.export)
env_spec_prms = modify_spec_prms(env_spec_prms, no_object=True)

export = sq_vis_dp_conv_1d.export.node_leaf_without_keys(
    ['dataset_train', 'dataset_holdout', 'env_spec', 'env_train']) & d(
    augment=False,
    batch_size=8,
    dataset='human_buds-kitchen_60k_eimgs',

    env_train=kitchen.export,
    env_spec=env_spec_prms,

    model=d(
        goal_names=[],
        state_names=['robot0_eef_pos', 'robot0_gripper_qpos'],
        vision_encoder=d(
            image_shape=[128, 128, 3],
            img_embed_size=128,
            # DP paper says this is important for EMA
            use_group_norm=True,
        ),
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=np_img_base_seq.export & d(
        load_episode_range=[0.0, 0.9],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/hvsBlock3D/{x}.npz'),
        batch_names_to_get=['robot0_eef_pos', 'robot0_gripper_qpos', 'action', 'image', 'ego_image'],
    ),
    dataset_holdout=np_img_base_seq.export & d(
        load_from_base=True,
        load_episode_range=[0.9, 1.0],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/hvsBlock3D/{x}.npz'),
        batch_names_to_get=['robot0_eef_pos', 'robot0_gripper_qpos', 'action', 'image', 'ego_image'],
    ),
    policy=d(
        online_action_postproc_fn=get_rs_online_action_postproc_fn(no_ori=True, fast_dynamics=True),
        is_terminated_fn=get_timeout_terminate_fn(1200),
    ),
    trainer=d(
        max_steps=1200000,
        rollout_train_env_every_n_steps=40000,
        save_checkpoint_every_n_steps=120000,
        train_do_data_augmentation=F('augment'),
    ),
)
