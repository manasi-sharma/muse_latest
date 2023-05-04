from attrdict import AttrDict as d

from cfgs.dataset import np_img_base_seq
from cfgs.env import push_t
from cfgs.exp_hvs.pusht import bc_rnn as pt_bc_rnn
from cfgs.model import vis_bc_rnn
from cfgs.model.vision import resnet_encoder
from configs.fields import Field as F


export = pt_bc_rnn.export.leaf_filter(lambda k, v: 'dataset' not in k) & d(
    batch_size=16,
    dataset='human_square_30k_eimgs',
    exp_name='push_t/posact_{?seed:s{seed}_}b{batch_size}_h{horizon}_{dataset}',
    env_train=push_t.export & d(imgs=True),
    # sequential dataset modifications (adding input file)
    model=vis_bc_rnn.export & d(
        state_names=['image'],
        vision_encoder=resnet_encoder.export & d(
            image_shape=[84, 84, 3],
            img_embed_size=64,
        ),
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=np_img_base_seq.export & d(
        use_rollout_steps=False,
        load_episode_range=[0.0, 0.9],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/push_t/{x}.npz'),
        batch_names_to_get=['robot0_eef_pos', 'robot0_gripper_qpos', 'action', 'image', 'ego_image'],
    ),
    dataset_holdout=np_img_base_seq.export & d(
        load_from_base=True,
        use_rollout_steps=False,
        load_episode_range=[0.9, 1.0],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/push_t/{x}.npz'),
        batch_names_to_get=['robot0_eef_pos', 'robot0_gripper_qpos', 'action', 'image', 'ego_image'],
    ),
    trainer=d(
        max_steps=600000,
    ),
)
