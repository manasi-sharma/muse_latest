"""
This file is very similar to eval_sequences. But enables adding / editing a mode key to the dataset

<controls here>

"""

import os
from configs.helpers import get_script_parser, load_base_config

import cv2
import numpy as np
from attrdict import AttrDict

from muse.datasets.np_dataset import NpDataset
from muse.experiments import logger
from muse.experiments.file_manager import ExperimentFileManager, FileManager
from muse.utils.file_utils import file_path_with_default_dir
from muse.utils.general_utils import exit_on_ctrl_c
from muse.utils.input_utils import query_string_from_set

if __name__ == '__main__':

    parser = get_script_parser()
    parser.add_argument('config', type=str)
    parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--load_file', type=str, default=None)
    parser.add_argument('--image_key', type=str, required=True)
    parser.add_argument('--click_key', type=str, default='click_state')
    parser.add_argument('--add_click_to_input', action='store_true')
    parser.add_argument('--flip_imgs', action="store_true")
    parser.add_argument('--y', action="store_true")
    parser.add_argument('--start_ep', type=int, default=0)  # which episode to start at

    # file_manager = ExperimentFileManager(params.exp_name, is_continue=True)
    args, unknown = parser.parse_known_args()
    # register_config_args(unknown)

    # load the config
    params, root = load_base_config(args.config, unknown)
    exp_name = root.get_exp_name()

    if os.path.exists(os.path.join(FileManager.base_dir, 'experiments', exp_name)):
        file_manager = ExperimentFileManager(exp_name, is_continue=True)
    else:
        logger.warn(f"Experiment: 'experiments/{exp_name}' does not exist, using 'experiments/test' instead")
        file_manager = ExperimentFileManager('test', is_continue=True)

    exit_on_ctrl_c()
    click_key = args.click_key

    if args.add_click_to_input:
        if click_key not in params.env_spec['action_names']:
            logger.warn("Adding clicks to input spec!")
            params.env_spec.action_names.extend([click_key])

    # if args.image_key not in params.env_spec.observation_names:
    #     params.env_spec.observation_names.append(args.image_key)
    env_spec = params.env_spec.cls(params.env_spec)

    # create dataset train (input)
    dataset_params = AttrDict(
        file=args.file,
        output_file='none.npz',
        batch_size=1,
        horizon=1,
        capacity=1e5,  # single episode, get_batch returns (1, H, ...)
    )
    dataset_input = NpDataset(dataset_params, env_spec, file_manager)

    env_spec_out_prms = params.env_spec.leaf_copy()

    # add click key to action names
    if click_key not in env_spec_out_prms['action_names']:
        logger.warn("Mode key not present in input. Will add to output")
        env_spec_out_prms.action_names.extend([click_key])
    env_spec_out = params.env_spec.cls(env_spec_out_prms)

    # create dataset out
    output_file = file_path_with_default_dir(args.output_file, file_manager.data_dir, expand_user=True)
    dataset_out_params = dataset_params.leaf_copy()

    if args.load_file is None:
        dataset_out_params.file = 'none.npz'  # no load
    else:
        dataset_out_params.file = args.load_file

    dataset_out_params.output_file = output_file
    dataset_out = NpDataset(dataset_out_params, env_spec_out, file_manager)

    eval_datadict = AttrDict()

    # where to start loading from in dataset_input
    start_ep = args.start_ep
    if start_ep == -1:
        start_ep = dataset_out.get_num_episodes()  # start at end of dataset_out (loaded)

    cv2.namedWindow("sequence_image", cv2.WINDOW_FULLSCREEN)
    running = True
    for ep in range(start_ep, dataset_input.get_num_episodes()):

        if not running:
            logger.warn(f"Exiting Early (at episode = {ep})!")
            break

        # (H x ...)
        inputs, outputs = dataset_input.get_episode(ep, names=None, split=True)
        meta = AttrDict()

        assert inputs.has_leaf_key(args.image_key), [inputs.list_leaf_keys(), args.image_key]

        batch_len = min(outputs.done.shape[0], inputs[args.image_key].shape[0])

        if not inputs.has_leaf_key(click_key):
            inputs[click_key] = np.zeros((batch_len, 1))  # match 1 x H x

        start_inputs = inputs.leaf_apply(lambda arr: arr[0])
        end_inputs = inputs.leaf_apply(lambda arr: arr[-1])

        i = 0
        while i < batch_len:
            logger.debug("Ep %d: [%d] press n = next, p = prev, q = next batch, esc = end.." % (ep, i))
            img = inputs[args.image_key][i]
            mode = inputs[click_key][i]
            if args.flip_imgs:
                img = img[..., ::-1]
            img = img.astype(np.uint8)

            # show the marker
            img = cv2.putText(img, f"M = {mode.item()}", (10, 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 255))

            cv2.imshow("sequence_image", img)
            # print("done:", outputs.done[0, i])
            # print("bp:", (inputs >> "block_positions")[0, i, 0])
            ret = cv2.waitKey(0)
            if ret == ord('m'):  # Mark.
                inputs[click_key][i] = 1
                if i < batch_len - 1:
                    i += 1  # stops once it reaches the last frame. only next will loop around.
            elif ret == 127:  # back space
                inputs[click_key][i] = 0
                if i > 0:
                    i -= 1
            elif ret == ord('f'):  # toggle
                inputs[click_key][i] = 1 - inputs[click_key][i]
            elif ret == ord('n'):  # next
                i = (i + 1) % batch_len
            elif ret == ord(']'):  # next double speed
                i = (i + 2) % batch_len
            elif ret == ord('p'):  # prev
                i = (i - 1) % batch_len
            elif ret == ord('['):  # prev double speed
                i = (i - 2) % batch_len
            elif ret == ord('q'):  # next batch
                i = batch_len
            elif ret == 27:  # esc
                i = batch_len
                running = False

        dataset_out.add_episode(inputs, outputs)

    do_save = args.y or query_string_from_set(f'Save to {dataset_out.save_dir}? (y/n)', ['y', 'n']) == 'y'
    if do_save:
        dataset_out.save()
