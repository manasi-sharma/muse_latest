from attrdict import AttrDict as d
from muse.policies.replay_policy import ReplayPolicy


export = d(
    cls=ReplayPolicy,
    demo_file="",  # fill this in
    ep_idx=0,
    action_names=['action'],
)
