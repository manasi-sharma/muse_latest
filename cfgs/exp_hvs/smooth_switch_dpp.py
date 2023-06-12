import numpy as np

from configs.fields import Field as F
from muse.datasets.preprocess.data_preprocessor import DataPreprocessor
from muse.experiments import logger
from attrdict import AttrDict as d


def get_switch_smooth_preproc_fn(switch_key, smooth_switch_key,
                                 kernel_size=5, temp=5,
                                 ramping_frac=None, min_ramping_len=4, debug=False):

    # kernel = kernel / np.sum(kernel)  # normalize

    kernel = np.exp(np.arange(kernel_size) / temp)
    kernel /= kernel[-1]

    if debug and ramping_frac is None:
        logger.debug(f"Using kernel: {kernel}")

    # default preproc: compute the mask, save to memory.
    def episode_preproc_fn(inputs, onetime_inputs, idx):
        inputs = inputs.leaf_copy()
        mode = inputs[switch_key]  # should be (H, 1) for same initial shape as the key

        mode = mode[..., 0].astype(float)  # remove last dim pre conv

        if ramping_frac is None:
            # pad the ends with the same (horizon dim)
            pad_mode = np.concatenate([mode[..., :1]] * (kernel_size - 1) + [mode], axis=-1)

            # [H,], [4*sn+1,] -> out shape: [H+4*sn,] -> [H,], float
            new_mode = np.convolve(pad_mode, kernel)[kernel_size - 1:, None]
        else:
            # ramp to 90%
            new_mode = mode.copy()
            end_idxs = np.flatnonzero(mode)
            start_idxs = np.concatenate([[0], end_idxs[:-1]])
            for s, e in zip(start_idxs, end_idxs):
                length = e - s
                n = max(int(np.ceil(length * ramping_frac)), min_ramping_len)
                # ramp to 90% in last <ttg_ramping_frac>% of the samples
                new_mode[e-n:e] = np.linspace(0, 0.9, n+1)[1:]

        # saturate at 1.
        mode = np.minimum(new_mode, 1)

        # smoothed key
        inputs[smooth_switch_key] = mode

        return inputs, onetime_inputs, [smooth_switch_key]

    return episode_preproc_fn


export = d(
    cls=DataPreprocessor,
    name="smooth_switch",
    episode_preproc_fn=get_switch_smooth_preproc_fn("switch", "smooth_switch", ramping_frac=0.2),
)
