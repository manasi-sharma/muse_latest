import numpy as np
import torch

from attrdict import AttrDict as d
from torch.nn import CrossEntropyLoss

from cfgs.dataset import np_seq
from cfgs.env import square
from cfgs.exp_hvs import smooth_cs_dpp
from cfgs.trainer import rm_goal_trainer
from configs.fields import Field as F, GroupField
from muse.datasets.preprocess.data_augmentation import DataAugmentation
from muse.envs.robosuite.robosuite_env import RobosuiteEnv
from muse.models.rnn_model import DefaultRnnModel
from muse.policies.basic_policy import BasicPolicy
from muse.utils.loss_utils import FocalLoss
from muse.utils.torch_utils import get_augment_fn


def click_spec_mod(params):
    env_spec = RobosuiteEnv.get_default_env_spec_params(params)
    env_spec.names_shapes_limits_dtypes.append(("smooth_click_state", (1,), (0, np.inf), np.float32))
    env_spec.action_names.extend(['mode', 'click_state'])
    return env_spec


def get_loss_fn(use_smooth_cs=False, loss_type='cross_entropy', focal_gamma=5, label_smoothing=0.):
    eval_obj = CrossEntropyLoss()
    if loss_type == 'cross_entropy':
        loss_obj = CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_type == 'focal':
        assert not use_smooth_cs, "smooth cs not compatible w/ focal loss!"
        loss_obj = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
    else:
        raise NotImplementedError

    def loss_fn(model, model_outputs, inputs, outputs, i=0, writer=None, writer_prefix="",
                ret_dict=False, meta=None, **kwargs):
        rcs = model_outputs['raw_click_state']
        B, H = rcs.shape[:2]

        if use_smooth_cs:
            smooth_cs = (inputs["smooth_click_state"]).view(-1, 1)
            target = torch.cat([1. - smooth_cs, smooth_cs], dim=-1)
        else:
            target = inputs['click_state'].to(dtype=torch.long).view(B * H)

        cs_loss = loss_obj(rcs.view(B * H, -1), target).mean()
        if writer is not None:
            eval_loss = eval_obj(rcs.view(B * H, -1), target).mean()
            writer.add_scalar(writer_prefix + f"click_state_loss", cs_loss.item(), i)
            writer.add_scalar(writer_prefix + f"click_state_eval_loss", eval_loss.item(), i)
        return cs_loss
    return loss_fn


def get_postproc_fn(online_thresh=0.5):
    def postproc_fn(inputs, model_outs):
        # P(cs=1) >= online_thresh
        model_outs['click_state'] = (model_outs['raw_click_state'][..., -1:] >= online_thresh).to(dtype=torch.long)
        return model_outs
    return postproc_fn

export = d(
    augment=False,
    device="cuda",
    batch_size=256,
    horizon=10,
    seed=0,
    load_frac=0.9,
    dataset='mode_real2_v2_fast_human_square_30k_imgs',
    exp_name='hvsBlock3D/cspred_{?seed:s{seed}_}{?augment:aug_}b{batch_size}_h{horizon}_{dataset}_lf{load_frac}',
    env_spec=GroupField('env_train', click_spec_mod),
    env_train=square.export,
    model=d(
        exp_name='_hs{hidden_size}{?use_smooth_cs:_scs}{?focal:_foc{focal_gamma}}{?label_smoothing:_ls{label_smoothing}}',
        cls=DefaultRnnModel,
        device=F('device'),
        horizon=F('horizon'),
        model_inputs=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        model_output='raw_click_state',
        out_size=2,
        hidden_size=400,

        # loss things
        use_smooth_cs=False,
        label_smoothing=0.,
        focal=False,
        focal_gamma=5,
        loss_fn=F(['use_smooth_cs', 'focal', 'focal_gamma', 'label_smoothing'],
                  lambda scs, f, fg, ls: get_loss_fn(use_smooth_cs=scs,
                                                     loss_type='focal' if f else 'cross_entropy',
                                                     focal_gamma=fg, label_smoothing=ls)),

        online_thresh=0.5,
        postproc_fn=F(['online_thresh'], lambda ot: get_postproc_fn(online_thresh=ot)),
    ),
    # sequential dataset modifications (adding input file)
    dataset_train=np_seq.export & d(
        load_episode_range=F('load_frac', lambda f: [0.0, f]),
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/hvsBlock3D/{x}.npz'),
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'object', 'robot0_gripper_qpos', 'policy_switch',
                            'robot0_eef_quat', 'action', 'mode', 'click_state', 'smooth_click_state'],
        data_preprocessors=[smooth_cs_dpp.export],
    ),
    dataset_holdout=np_seq.export & d(
        load_episode_range=F('load_frac', lambda f: [f, 1.0]),
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/hvsBlock3D/{x}.npz'),
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'object', 'robot0_gripper_qpos', 'policy_switch',
                            'robot0_eef_quat', 'action', 'mode', 'click_state', 'smooth_click_state'],
        data_preprocessors=[smooth_cs_dpp.export],
    ),

    policy=d(
        cls=BasicPolicy,
        policy_model_forward_fn=lambda m, o, g, **kwargs: d(),
        timeout=2,
    ),
    goal_policy=d(
        cls=BasicPolicy,
        policy_model_forward_fn=lambda m, o, g, **kwargs: d(),
        timeout=2,
    ),
    trainer=rm_goal_trainer.export & d(
        train_do_data_augmentation=F('augment'),
        rollout_train_env_every_n_steps=0,
        track_best_name=None,
        save_checkpoint_every_n_steps=20000,

        data_augmentation_params=d(
            cls=DataAugmentation,
            augment_keys=['robot0_eef_pos', 'robot0_eef_quat'],
            augment_fns=[get_augment_fn(0.02), get_augment_fn(0.04)],
        ),
    ),
)
