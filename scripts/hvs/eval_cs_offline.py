
import os
import sys

import math

import cv2
import imageio
import numpy as np
import torch
from typing import List, Tuple

from attrdict import AttrDict as d
from torch.nn import functional as F

from configs.helpers import load_base_config, get_script_parser
from muse.experiments import logger
from muse.experiments.file_manager import ExperimentFileManager
from muse.utils.file_utils import file_path_with_default_dir
from muse.utils.general_utils import exit_on_ctrl_c, get_with_default
from muse.utils.torch_utils import get_horizon_chunks, to_torch, to_numpy

if __name__ == '__main__':

    parser = get_script_parser()
    parser.add_argument('config', type=str)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
    parser.add_argument('--load_range', type=float, nargs=2, default=[0., 1.], help="lower and upper frac")
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--no_model_file', action="store_true")
    parser.add_argument('--eval_batch_size', type=int, default=None)  # ignore for now
    parser.add_argument('--eval_horizon', type=int, default=0)  # if 0, use data train's horizon
    parser.add_argument('--do_model_forward', action="store_true")
    parser.add_argument('--do_loss', action="store_true")
    parser.add_argument('--show_prob', action="store_true", help='Show the probability instead of argmax')
    parser.add_argument('--raw_key', type=str, default='raw_click_state', help='raw key')
    parser.add_argument('--skip_video', action="store_true", )
    # if horizon > 1, eval len will be different than data len, pad eval with zeros
    parser.add_argument('--pad_eval_zeros', action="store_true")
    local_args, unknown = parser.parse_known_args()

    # load the config
    params, root = load_base_config(local_args.config, unknown)
    exp_name = root.get_exp_name()

    file_manager = ExperimentFileManager(exp_name, is_continue=True)

    if local_args.model_file is not None:
        model_fname = local_args.model_file
        model_fname = file_path_with_default_dir(model_fname, file_manager.models_dir)
        assert os.path.exists(model_fname), 'Model: {0} does not exist'.format(model_fname)
        logger.debug("Using model: {}".format(model_fname))
    else:
        model_fname = os.path.join(file_manager.models_dir, "model.pt")
        logger.debug("Using default model for current eval: {}".format(model_fname))

    # generate env
    env_spec = params.env_spec.cls(params.env_spec)

    # generate model and policy
    model = params.model.cls(params.model, env_spec, None)

    if local_args.eval_horizon > 0:
        HORIZON = local_args.eval_horizon
        logger.warn("Overriding horizon in dset with %d" % HORIZON)
    else:
        HORIZON = get_with_default(params.dataset_train, "horizon", 0)

    if local_args.eval_batch_size is None:
        local_args.eval_batch_size = params['batch_size']

    assert HORIZON > 0
    params.dataset_train.file = local_args.file
    params.dataset_train.load_episode_range = local_args.load_range
    params.dataset_train.horizon = HORIZON
    dataset_input = params.dataset_train.cls(params.dataset_train, env_spec, file_manager)

    output_file = file_path_with_default_dir(local_args.output_file, file_manager.exp_dir, expand_user=True)

    ### restore model
    if not local_args.no_model_file:
        model.restore_from_file(model_fname)

    all_names = env_spec.all_names + ["done", "rollout_timestep"]

    eval_datadict = []


    def get_concat_if_concatable(dim):
        def concat_fn(vs):
            if isinstance(vs[0], np.ndarray):
                return np.concatenate(vs, axis=dim)
            elif isinstance(vs[0], torch.Tensor):
                return torch.cat(vs, dim=dim)
            else:
                return vs
        return concat_fn


    for i in range(dataset_input.get_num_episodes()):
        # (N x ...) where N is episode length
        inputs, outputs = dataset_input.get_episode(i, all_names, split=True)
        # print(datadict.done.shape, HORIZON)

        if len(outputs.done) < HORIZON:
            logger.warn("Episode %d does not have enough data pts (%d instead of %d). Skipping."
                        % (i, len(outputs.done), HORIZON))
            continue

        # # (inclusive range), new shape is (N - H + 1, H, ...)
        inputs_chunked = inputs\
            .leaf_apply(lambda arr: get_horizon_chunks(arr, HORIZON, 0, len(outputs.done) - HORIZON,
                                                       dim=0, stack_dim=0))
        outputs_chunked = outputs\
            .leaf_apply(lambda arr: get_horizon_chunks(arr, HORIZON, 0, len(outputs.done) - HORIZON,
                                                       dim=0, stack_dim=0))

        # model.eval()
        with torch.no_grad():
            model_forward = []
            losses = []

            num_chunks = len(outputs_chunked.done)
            batch_size = min(max(local_args.eval_batch_size, 1), num_chunks)
            num_batches = math.ceil(num_chunks / batch_size)
            for j in range(num_batches):
                start, end = j * batch_size, min((j+1) * batch_size, num_chunks)
                # convert to torch
                inputs_t = inputs_chunked.leaf_filter(lambda k, v: v.dtype != object)\
                    .leaf_apply(lambda arr: to_torch(arr, device=model.device))
                outputs_t = outputs_chunked.leaf_filter(lambda k, v: v.dtype != object)\
                    .leaf_apply(lambda arr: to_torch(arr, device=model.device))

                # for policy (np)
                inputs_j = inputs_chunked.leaf_apply(lambda arr: arr[start:end])
                goals_j = inputs_chunked.leaf_apply(lambda arr: arr[start:end])
                outputs_j = outputs_chunked.leaf_apply(lambda arr: arr[start:end])

                # for model (torch)
                inputs_t_j = inputs_t.leaf_apply(lambda arr: arr[start:end])
                outputs_t_j = inputs_t.leaf_apply(lambda arr: arr[start:end])

                model_forward_j = d()
                if local_args.do_model_forward:
                    # hacky to remove big arrays
                    model_forward_j = model.forward(inputs_t_j).leaf_filter(lambda k,v: 'policy' not in k)

                loss_j = d()
                if local_args.do_loss:
                    loss_out = model.loss(inputs_t_j, outputs_t_j, ret_dict=False)
                    if isinstance(loss_out, List) or isinstance(loss_out, Tuple):
                        for j in range(len(loss_out)):
                            loss_j["loss_%d" % j] = loss_out[j]
                    elif isinstance(loss_out, d):
                        loss_j = loss_out
                    else:
                        loss_j.loss = loss_out[None]  # single loss

                model_forward.append(model_forward_j.leaf_arrays().leaf_apply(lambda arr: to_numpy(arr, check=True)))
                losses.append(loss_j.leaf_arrays().leaf_apply(lambda arr: to_numpy(arr, check=True)))

            # evaluations for this episode
            model_forward = d.leaf_combine_and_apply(model_forward, get_concat_if_concatable(0))
            losses = d.leaf_combine_and_apply(losses, get_concat_if_concatable(0))

            # def to_np_pad(arr):
            #     arr = to_numpy(arr, check=True)
            #     # do this to make sure everything is aligned and the same length
            #     if args.pad_eval_zeros:
            #         padding = [[0, 0] for _ in range(len(arr.shape))]
            #         padding[0][1] = len(outputs.done) - num_chunks
            #         return np.pad(arr, padding)
            #     return arr

            evaluations = d()
            evaluations.model = model_forward  #.leaf_apply(lambda arr: to_np_pad(arr) if isinstance(arr, torch.Tensor) else arr)
            evaluations.losses = losses  #.leaf_apply(lambda arr: to_np_pad(arr) if isinstance(arr, torch.Tensor) else arr)

        # special processing for click state (last horizon element of the batch except for first H-1)
        # (H, ...) cat (N - H, ...)
        evaluations.click_state = np.concatenate([evaluations.model.click_state[0],
                                                  evaluations.model.click_state[1:, -1]], axis=0)

        new_datadict = d()
        new_datadict.combine(inputs)
        new_datadict.combine(outputs)
        new_datadict.evaluations = evaluations


        # print(obs.ee_position.shape, new_action.action.shape, out_obs.next_ee_position.shape)

        eval_datadict.append(new_datadict)
        if i == 0:
            logger.info("Using keys for first batch: " + str(list(new_datadict.leaf_keys())))
            for key, item in new_datadict.leaf_items():
                logger.debug("%s : type: %s, shape: %s" % (key, type(item), item.shape if isinstance(item, np.ndarray) else []))

    if not local_args.skip_video:
        imgs = []
        # SUPER hacky
        for i in range(dataset_input.get_num_episodes()):
            ep = dataset_input.get_episode(i, ['image', 'click_state'], split=False)
            # execute last horizon element per window
            for j in range(len(ep.image)):
                cs = ep['click_state'][j].item()
                # last horizon element.
                idx = max(j - HORIZON, 0)
                spill = min(j, HORIZON - 1)
                if local_args.show_prob:
                    probs = eval_datadict[i][f'evaluations/model/{local_args.raw_key}'][idx, spill]
                    if local_args.raw_key == 'raw_click_state':
                        pred_cs = F.softmax(torch.from_numpy(probs), dim=-1)[..., 1].numpy()  # probability
                    elif local_args.raw_key == 'raw_mode_switch':
                        pred_mode_switch = F.sigmoid(torch.from_numpy(probs)).numpy()  # probability
                        # either mode is 1 or termination is 1
                        pred_cs = [pred_mode_switch[..., :1].item(), pred_mode_switch[..., 1:].item()]
                    else:
                        raise NotImplementedError(local_args.raw_key)
                else:
                    pred_cs = eval_datadict[i]['evaluations/model/click_state'][idx, spill].item()
                ep.image[j] = ep.image[j].astype(np.uint8)
                # show the marker in magenta at the top...
                if isinstance(pred_cs, list):
                    ep.image[j] = cv2.putText(ep.image[j], f"CS={cs}", (5, 10), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 0, 255))
                    ep.image[j] = cv2.putText(ep.image[j], f"P ={pred_cs[0]:.1f}/{pred_cs[1]:.1f}", (5, 20), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 0, 255))
                else:
                    ep.image[j] = cv2.putText(ep.image[j], f"CS={cs}/{pred_cs:.1f}", (5, 10), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 0, 255))
            imgs.append(ep.image)

    eval_datadict = d.leaf_combine_and_apply(eval_datadict, lambda vs: np.concatenate(vs, axis=0))

    logger.debug("Saving dataset output to -> %s" % output_file)

    logger.debug("Keys: " + str(list(eval_datadict.leaf_keys())))
    to_save = dict()
    for name in eval_datadict.leaf_keys():
        to_save[name] = eval_datadict[name]

    np.savez_compressed(output_file, **to_save)

    if not local_args.skip_video:
        # TODO save video optionally
        video_file = output_file[:-4] + ".mp4"

        # image saving
        imgs = np.concatenate(imgs, axis=0)  # (H x ...)

        logger.debug(f"Images output shape: {imgs.shape}")

        # saving images
        logger.debug("Saving video of length %d, fps %d to file -> %s" % (len(imgs), 10, video_file))

        # if args.raw:
        #     postprocess = lambda x: x
        # else:
        #     postprocess = lambda x: np.flip(x, axis=-1)  # BGR -> RGB
        # imgs = postprocess(imgs)

        imageio.mimsave(video_file, imgs.astype(np.uint8), format='mp4', fps=10)

        logger.debug("Saved.")
