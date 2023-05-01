from attrdict import AttrDict as d

from muse.trainers.stabilizers import EMA

export = d(
    cls=EMA,
    update_after_step=0,
    inv_gamma=1.0,
    power=0.75,
    min_value=0.0,
    max_value=0.9999,
)
