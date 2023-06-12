from attrdict import AttrDict as d

from cfgs.dataset import np_seq
from cfgs.env import fixed_pm
from cfgs.model import bc_mlp

from cfgs.trainer import rm_goal_trainer
from configs.fields import Field as F, GroupField
from muse.envs.simple.point_mass_env import get_online_action_postproc_fn, PointMassEnv
from muse.policies.basic_policy import BasicPolicy
from muse.policies.bc.gcbc_policy import GCBCPolicy
from muse.policies.memory_policy import get_timeout_terminate_fn

export = d(
    device="cuda",
    batch_size=256,
    horizon=10,
    seed=0,
    load_eps=0,  # set this to set the number of episodes to load from dataset (first X)
    dataset='pm_direct_1000ep',
    exp_name='pmhvs/velact_{?seed:s{seed}_}b{batch_size}_h{horizon}_{dataset}{?load_eps:-first{load_eps}}',
    # utils=utils,
    env_spec=GroupField('env_train', PointMassEnv.get_default_env_spec_params),
    env_train=fixed_pm.export,

    # True = use of fixed scale norms (1 for pos, 10 for ori, and 100 for gripper)
    model=bc_mlp.export & d(
        goal_names=[],
        state_names=['ego', 'target'],
        device=F('device'),
        action_decoder=d(
            mlp_size=200,
        )
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=np_seq.export & d(
        initial_load_episodes=F('load_eps'),
        load_episode_range=[0.0, 0.9],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/pmhvs/{x}.npz'),
        batch_names_to_get=['ego', 'target', 'action'],
    ),
    dataset_holdout=np_seq.export & d(
        initial_load_episodes=F('load_eps'),
        load_episode_range=[0.9, 1.0],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/pmhvs/{x}.npz'),
        batch_names_to_get=['ego', 'target', 'action'],
    ),

    policy=d(
        cls=GCBCPolicy,
        velact=True,
        policy_out_names=['action'],
        policy_out_norm_names=[],
        fill_extra_policy_names=True,
        # shared online function with real environment
        online_action_postproc_fn=get_online_action_postproc_fn(),
        is_terminated_fn=get_timeout_terminate_fn(500),  # episode length online
    ),
    goal_policy=d(
        cls=BasicPolicy,
        policy_model_forward_fn=lambda m, o, g, **kwargs: d(),
        timeout=2,
    ),
    # same evaluation scheme as robomimic environments, but less evals cuz its slower
    trainer=rm_goal_trainer.export & d(
        max_steps=200000,
        block_env_on_first_n_steps=10000,
        rollout_train_env_every_n_steps=10000,
        save_every_n_steps=10000,
    ),
)
