from attrdict import AttrDict as d

from cfgs.exp_hvs.coffee import vis_bc_rnn as coffee_vis_bc_rnn

export = coffee_vis_bc_rnn.export & d(
    dataset='raw_mode_b2b_eimgs_20ep',
)
