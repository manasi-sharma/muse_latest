import numpy as np
import torch

from attrdict import AttrDict as d
from torch.nn import BCEWithLogitsLoss

from cfgs.dataset import np_seq
from cfgs.env import square
from cfgs.exp_hvs import parse_switch_dpp, smooth_switch_dpp, smooth_dpp
from cfgs.trainer import rm_goal_trainer
from configs.fields import Field as F, GroupField
from muse.datasets.preprocess.data_augmentation import DataAugmentation
from muse.envs.robosuite.robosuite_env import RobosuiteEnv
from muse.models.rnn_model import DefaultRnnModel
from muse.policies.basic_policy import BasicPolicy
from muse.utils.torch_utils import get_augment_fn


def click_spec_mod(params):
    env_spec = RobosuiteEnv.get_default_env_spec_params(params)
    env_spec.names_shapes_limits_dtypes.append(("smooth_mode", (1,), (0, np.inf), np.float32))
    env_spec.names_shapes_limits_dtypes.append(("switch", (1,), (0, np.inf), np.float32))
    env_spec.names_shapes_limits_dtypes.append(("smooth_switch", (1,), (0, np.inf), np.float32))
    env_spec.action_names.extend(['mode', 'click_state'])
    return env_spec


def write_accuracy(logits, true, writer, writer_prefix, i, thresh=0.5):
    pred = (torch.sigmoid(logits.reshape(-1)) >= thresh)
    TP = (pred & true).sum().item()
    TN = ((~pred) & (~true)).sum().item()
    FP = ((pred) & (~true)).sum().item()
    FN = ((~pred) & (true)).sum().item()
    TOTAL = pred.numel()

    writer.add_scalar(writer_prefix + f"tpr(recall)", TP / (TP + FN + 1e-11), i)
    writer.add_scalar(writer_prefix + f"tnr", TN / (TN + FP + 1e-11), i)
    writer.add_scalar(writer_prefix + f"ppv(precision)", TP / (TP + FP + 1e-11), i)
    writer.add_scalar(writer_prefix + f"accuracy", (TP + TN) / TOTAL, i)


def get_loss_fn(use_smooth_mode=False, use_smooth_switch=False, beta=1., mode_thresh=0.5, switch_thresh=0.5):
    loss_obj = BCEWithLogitsLoss()

    def loss_fn(model, model_outputs, inputs, outputs, i=0, writer=None, writer_prefix="",
                ret_dict=False, meta=None, **kwargs):
        mode_logits = model_outputs['raw_mode_switch'][..., :1]
        switch_logits = model_outputs['raw_mode_switch'][..., 1:]
        B, H = mode_logits.shape[:2]

        target_mode = inputs['smooth_mode' if use_smooth_mode else 'mode'].view(B * H)
        target_switch = inputs['smooth_switch' if use_smooth_switch else 'switch'].view(B * H)

        mode_loss = loss_obj(mode_logits.view(B * H), target_mode).mean()
        switch_loss = loss_obj(switch_logits.view(B * H), target_switch).mean()
        loss = mode_loss + beta * switch_loss

        if writer is not None:
            writer.add_scalar(writer_prefix + f"loss", loss.item(), i)
            writer.add_scalar(writer_prefix + f"mode/loss", mode_loss.item(), i)
            writer.add_scalar(writer_prefix + f"switch/loss", switch_loss.item(), i)
            write_accuracy(mode_logits, inputs['mode'].view(B*H) > 0.5, writer, writer_prefix+"mode/", i, thresh=mode_thresh)
            write_accuracy(switch_logits, inputs['switch'].view(B*H) > 0.5, writer, writer_prefix+"switch/", i, thresh=switch_thresh)
        return loss

    return loss_fn


def get_postproc_fn(mode_thresh=0.5, switch_thresh=0.9):
    def postproc_fn(inputs, model_outs):
        # if either is greater than its thresh
        ms = model_outs['raw_mode_switch']
        model_outs['click_state'] = torch.logical_or(ms[..., :1] >= mode_thresh,
                                                     ms[..., 1:] >= switch_thresh).to(dtype=torch.long)
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
    exp_name='hvsBlock3D/mspred_{?seed:s{seed}_}{?augment:aug_}b{batch_size}_h{horizon}_{dataset}_lf{load_frac}',
    env_spec=GroupField('env_train', click_spec_mod),
    env_train=square.export,
    model=d(
        exp_name='{?bidirectional:_bi}_hs{hidden_size}{?use_smooth_mode:_sm}{?use_smooth_switch:_ss}',
        cls=DefaultRnnModel,
        device=F('device'),
        horizon=F('horizon'),
        model_inputs=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        model_output='raw_mode_switch',
        bidirectional=False,
        out_size=2,
        hidden_size=400,

        # loss things
        use_smooth_mode=False,
        use_smooth_switch=True,
        mode_thresh=0.5,
        switch_thresh=0.9,
        loss_fn=F(['use_smooth_mode', 'use_smooth_switch', 'mode_thresh', 'switch_thresh'],
                  lambda sm, ss, mt, st: get_loss_fn(use_smooth_mode=sm, use_smooth_switch=ss,
                                                     mode_thresh=mt, switch_thresh=st)),

        postproc_fn=F(['mode_thresh', 'switch_thresh'], lambda mt, st: get_postproc_fn(mode_thresh=mt, switch_thresh=st)),
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=np_seq.export & d(
        load_episode_range=F('load_frac', lambda f: [0.0, f]),
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/hvsBlock3D/{x}.npz'),
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'object', 'robot0_gripper_qpos', 'policy_switch',
                            'robot0_eef_quat', 'action', 'mode', 'click_state', 'switch', 'smooth_mode', 'smooth_switch'],
        data_preprocessors=[parse_switch_dpp.export, smooth_switch_dpp.export, smooth_dpp.export]
    ),
    dataset_holdout=np_seq.export & d(
        load_episode_range=F('load_frac', lambda f: [f, 1.0]),
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/hvsBlock3D/{x}.npz'),
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'object', 'robot0_gripper_qpos', 'policy_switch',
                            'robot0_eef_quat', 'action', 'mode', 'click_state', 'switch', 'smooth_mode', 'smooth_switch'],
        data_preprocessors=[parse_switch_dpp.export, smooth_switch_dpp.export, smooth_dpp.export],
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
        max_steps=600000,
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
