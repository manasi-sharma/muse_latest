import numpy as np
from attrdict import AttrDict as d

from cfgs.dataset import np_seq
from cfgs.env import block3d
from cfgs.model import bc_rnn

from cfgs.trainer import rm_goal_trainer
from configs.fields import Field as F, GroupField
from muse.envs.polymetis.polymetis_utils import get_polymetis_online_action_postproc_fn
from muse.policies.basic_policy import BasicPolicy
from muse.policies.bc.gcbc_policy import GCBCPolicy
from muse.policies.memory_policy import get_timeout_terminate_fn
from muse.utils.loss_utils import get_default_nll_loss_fn, get_default_mae_action_loss_fn, mse_err_fn

export = d(
    device="cuda",
    batch_size=256,
    horizon=10,
    seed=0,
    dataset='lift_diverse_delta_500ep_eimgs',
    exp_name='affordance/velact_{?seed:s{seed}_}b{batch_size}_h{horizon}_{dataset}{?use_pose_norm:_pn}',
    # utils=utils,
    env_spec=GroupField('env_train', block3d.export.cls.get_default_env_spec_params),
    env_train=block3d.export & d(task_name='lift', compute_images=False),

    # True = use of fixed scale norms (1 for pos, 10 for ori, and 100 for gripper)
    use_pose_norm=False,
    model=bc_rnn.export & d(
        goal_names=['objects/position', 'objects/orientation'],
        state_names=['ee_position', 'ee_orientation', 'gripper_pos', 'objects/position', 'objects/orientation'],
        device=F('device'),
        horizon=F('horizon'),
        # scales for loss (on delta action, delta ori is 5x, gripper is absolute -1 -> 1)
        norm_overrides=F('use_pose_norm', lambda pn: (d(action=d(mean=np.zeros(7),
                                                                 std=np.array([1., 1., 1., 5., 5., 5., 50.])))
                                                      if pn else d())),
        save_action_normalization=True,
        loss_fn=F('use_policy_dist',
                  lambda x: get_default_nll_loss_fn(['action'], policy_out_norm_names=['action'], vel_act=True)
                  if x else
                  get_default_mae_action_loss_fn(['action'], max_grab=None,
                                                 err_fn=mse_err_fn, vel_act=True,
                                                 policy_out_norm_names=['action'])
                  ),
        action_decoder=d(
            use_tanh_out=False,  # actions are not -1 to 1
        ),
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=np_seq.export & d(
        load_episode_range=[0.0, 0.9],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/affordance/{x}.npz'),
        batch_names_to_get=['ee_position', 'ee_orientation', 'objects/position',
                            'objects/orientation', 'gripper_pos', 'action'],
    ),
    dataset_holdout=np_seq.export & d(
        load_episode_range=[0.9, 1.0],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/affordance/{x}.npz'),
        batch_names_to_get=['ee_position', 'ee_orientation', 'objects/position',
                            'objects/orientation', 'gripper_pos', 'action'],
    ),

    policy=d(
        cls=GCBCPolicy,
        velact=True,
        recurrent=True,
        policy_out_names=['action'],
        policy_out_norm_names=['action'],
        fill_extra_policy_names=True,
        # shared online function with real environment
        online_action_postproc_fn=get_polymetis_online_action_postproc_fn(no_ori=False, fast_dynamics=True),
        is_terminated_fn=get_timeout_terminate_fn(120),  # episode length online
    ),
    goal_policy=d(
        cls=BasicPolicy,
        policy_model_forward_fn=lambda m, o, g, **kwargs: d(),
        timeout=2,
    ),
    # same evaluation scheme as robomimic environments, but less evals cuz its slower
    trainer=rm_goal_trainer.export & d(
        rollout_train_env_every_n_steps=50000,
        save_every_n_steps=50000,
    ),
)
