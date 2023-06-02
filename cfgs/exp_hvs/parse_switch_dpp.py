import numpy as np
import torch

from muse.datasets.preprocess.data_preprocessor import DataPreprocessor
from attrdict import AttrDict as d


def get_parse_switch_preproc_fn(click_state_key, switch_key, time_to_go_key=None):

    # default preproc: compute the mask, save to memory.
    def episode_preproc_fn(inputs, onetime_inputs, idx):
        inputs = inputs.leaf_copy()
        cs = inputs[click_state_key]  # should be (H, 1) for same initial shape as the key
        next_cs = np.concatenate([cs[1:], cs[-1:]], axis=0).astype(bool)
        prev_cs = np.concatenate([cs[:1], cs[:-1]], axis=0).astype(bool)

        # switch is.. (prev, curr, next)
        #             (0 1 0)
        #             (0 1 1) (I know this is off by one but its chill)
        #             (1 1 0)

        inputs[switch_key] = (cs & ~(next_cs & prev_cs)).astype(int)
        inputs[switch_key][-1] = 1
        end_indices = np.flatnonzero(inputs[switch_key].reshape(-1))

        changed = [switch_key]
        if time_to_go_key is not None:
            changed.append(time_to_go_key)
            last = 0
            to_go = np.zeros_like(inputs[switch_key])
            for idx in end_indices:
                to_go[last:idx+1] = np.arange(idx + 1 - last)[::-1]
                last = idx + 1
            inputs[time_to_go_key] = to_go

        return inputs, onetime_inputs, changed

    return episode_preproc_fn


export = d(
    cls=DataPreprocessor,
    name="parse_switch_dpp",
    episode_preproc_fn=get_parse_switch_preproc_fn("click_state", "switch"),
)
