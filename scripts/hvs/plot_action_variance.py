"""
2D action variance through KNN
"""

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster
from attrdict import AttrDict

from muse.experiments import logger
from muse.experiments.file_manager import FileManager
from muse.utils import plt_utils
from muse.utils.file_utils import file_path_with_default_dir
from muse.utils.general_utils import exit_on_ctrl_c
from muse.utils.np_utils import np_split_dataset_by_key
from muse.utils.torch_utils import combine_dims_np, split_dim_np, combine_after_dim

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
parser.add_argument('--key', type=str, help="Key in dataset to cluster by")
parser.add_argument('--action_key', type=str, help="Key in dataset to compute variance over")
parser.add_argument('--save_file', type=str, default=None, help="Save animation here")
parser.add_argument('--plot_axes', type=int, nargs='*', default=None,
                    help="2-3 idxs in data to correspond with x,y... one indexed. negative to scale by -1.")
parser.add_argument('--temporal_horizon', type=int, default=None,
                    help='If not None, will do K means on H states instead of just 1.')
parser.add_argument('--temporal_idx', type=int, default=None, help='If not None, will index dim=1 with this idx')
parser.add_argument('--done_key', type=str, default='done')
parser.add_argument('--ep_range', type=int, nargs='*', default=None, help="episodes to run, range")  # TODO
parser.add_argument('--ep_idxs', type=int, nargs="*", default=[],
                    help="episodes to run, [] for all. done must be specified")  # TODO
# parser.add_argument('--ep_all_together', type=int, nargs="*", default=[], help="episodes to plot, [] for all")
parser.add_argument('--do_3d', action="store_true", help="do 3d version. not implemented yet")
parser.add_argument('--show', action="store_true", help="show plots")
parser.add_argument('--scatter', action="store_true", help="scatter instead of line")
parser.add_argument('--scale_minmax', action="store_true", help="x,y axes will be scaled to min/max.")
parser.add_argument('--clip_x', type=int, nargs=2, default=None)
parser.add_argument('--clip_y', type=int, nargs=2, default=None)
parser.add_argument('--cluster_frac', type=float, default=None, help='TODO Use this % of data nearest states as the basis')
parser.add_argument('--cluster_eps', type=float, default=None, help='If |s - s\'| < eps use s\' to compute basis')
parser.add_argument('--max_var', type=float, default=None, help='use a constant max var for cmap')
parser.add_argument('--draw_action_arrows', action='store_true', help='draw the cluster action(s)')
args = parser.parse_args()

exit_on_ctrl_c()  # in case of infinite waiting

assert len(args.file) >= 1

data = None

for f in args.file:
    path = file_path_with_default_dir(f, FileManager.base_dir, expand_user=True)
    logger.debug("File Path: %s" % path)
    new_data = AttrDict.from_dict(dict(np.load(path, allow_pickle=True)))
    if data is None:
        data = new_data
    else:
        common_keys = set(data.list_leaf_keys()).intersection(data.list_leaf_keys())
        data = AttrDict.leaf_combine_and_apply([data > common_keys, new_data > common_keys],
                                               lambda vs: np.concatenate(vs, axis=0))

logger.debug(list(data.leaf_keys()))

DO_3D = args.do_3d
default_plt_axes = [1, 2, 3] if DO_3D else [1, 2]
if args.plot_axes is None:
    args.plot_axes = default_plt_axes

plotting_axes = [abs(i) - 1 for i in args.plot_axes]
sgns = [int(i > 0) * 2 - 1 for i in args.plot_axes]

assert len(plotting_axes) == 3 if DO_3D else len(plotting_axes) == 2, "Wrong num axes!"

key = args.key
action_key = args.action_key
done_key = args.done_key
logger.debug("Key Name: %s, raw shape: %s" % (key, data[key].shape))
logger.debug("Action Key Name: %s, raw shape: %s" % (key, data[action_key].shape))
logger.debug("Episode Done Name: %s" % done_key)

to_save = {}
# split by key, then save each ep
done = data[done_key]
splits, data_ls, _ = np_split_dataset_by_key(data > [key, action_key], AttrDict(), done, complete=True)
if args.ep_range is not None:
    if len(args.ep_range) == 1:
        args.ep_range = [0, args.ep_range[0]]
    else:
        assert len(args.ep_range) == 2
    assert args.ep_range[0] < args.ep_range[1] <= len(splits), f"Ep range is invalid: {args.ep_range}"
    ep_idxs = list(range(args.ep_range[0], args.ep_range[1]))
else:
    ep_idxs = list(range(len(splits))) if len(args.ep_idxs) == 0 else args.ep_idxs

if args.temporal_horizon is not None:
    TH = args.temporal_horizon
    STATE_DIM = (data_ls[0] >> key).shape[-1]
    assert TH > 1, "temporal horizon must be at least 2 if specified..."
    logger.debug(f"Stacking states for {key} by temporal horizon {TH}")
    for ep in ep_idxs:
        arr = data_ls[ep][key]
        # pad
        arr = np.concatenate([arr[:1]] * (TH - 1) + [arr], axis=0)
        # combine dims 1&2, e.g., if arr is (B,2), then (B, TH*2) will be output
        data_ls[ep][key] = combine_dims_np(get_horizon_chunks(arr, TH, 0, len(arr) - TH, dim=0, stack_dim=0), 1)

# all of them
all_data_key = np.concatenate([data_ls[ep][key] for ep in ep_idxs], axis=0)
action_data = np.concatenate([data_ls[ep][action_key] for ep in ep_idxs], axis=0)
N = action_data.shape[0]

logger.debug("New Data shape: %s" % str(all_data_key.shape))
logger.debug("New Action shape: %s" % str(action_data.shape))

if args.cluster_eps is not None:
    eps = args.cluster_eps
    # N x N x D
    diff = combine_after_dim(all_data_key[None] - all_data_key[:, None], 2)
    dist_mat = np.linalg.norm(diff, axis=-1)
    # N x N bool
    cluster_sources = dist_mat <= eps
    cluster_bin_sizes = cluster_sources.sum(-1)
else:
    raise NotImplementedError('only epsilon based clustering is implemented')

nd = action_data.shape[-1]

assert len(action_data.shape) == 2
each_var = np.zeros((N, nd))

for i in range(N):
    action = action_data[i]
    cluster_source = cluster_sources[i]
    mean_action = action_data[cluster_source].mean(axis=0)

    each_var[i] = (action - mean_action) ** 2

# average to get the variance (uniformly likely samples
var = each_var.mean(axis=0)

logger.debug('-----------------------------------------------------------')

logger.debug(f'Files = {args.file}')
for dim in range(nd):
    logger.debug(f'Var[{dim}] = {var[dim]}')
logger.debug(f'Total Var = {var.sum()}')

logger.debug(f'Bin size, avg={cluster_bin_sizes.mean()}, med={np.median(cluster_bin_sizes)}, '
             f'range=[{np.min(cluster_bin_sizes)}, {np.max(cluster_bin_sizes)}]')

logger.debug('-----------------------------------------------------------')

def get_plotting_state(code):
    # re extract only the "state", e.g. the last state (corresponding to current time step)
    return split_dim_np(code, -1, [TH, STATE_DIM])[..., -1, :] if args.temporal_horizon is not None else code

#
# if args.temporal_idx is None:
#     xy_data = get_plotting_state(all_data_key)[..., plotting_axes]
# else:
#     xy_data = get_plotting_state(all_data_key)[:, args.temporal_idx, ..., plotting_axes]
#     action_data = action_data[:, args.temporal_idx]
#
# xy_data = xy_data.reshape(-1, len(plotting_axes))

# fig, axes = plt.subplots(num_elements_per_key, 1, figsize=(10, 4*num_elements_per_key))
fig = plt.figure(figsize=(nd * 5, 6), tight_layout=True)
gs = fig.add_gridspec(6, nd)
axes = [[fig.add_subplot(gs[:5, i], projection='3d' if DO_3D else None), fig.add_subplot(gs[5, i])] for i in range(nd)]
# axes = fig.subplots(1, nd, subplot_kw={'projection': '3d'} if DO_3D else None)
fig.suptitle(f"Var in action = {action_key}, given state = {key}" + (
    f" (h_idx={args.temporal_idx})" if args.temporal_idx is not None else ""))

cmap = mpl.cm.get_cmap('Oranges_r')

for dim in range(var.shape[-1]):
    dim_var = each_var[:, dim]

    ax_top = axes[dim][0]
    ax_bot = axes[dim][1]

    min_var = 0 if args.max_var is not None else np.min(dim_var)
    max_var = args.max_var if args.max_var is not None else np.max(dim_var)

    colors = [cmap((av - min_var) / (max_var - min_var)) for av in dim_var]

    codebook_plot = get_plotting_state(all_data_key)

    codebook_axes = np.split(codebook_plot[:, plotting_axes], len(plotting_axes), axis=-1)

    if args.draw_action_arrows:
        raise NotImplementedError
        for i, action in enumerate(means):
            ax_top.arrow(codebook_axes[0][i, 0], codebook_axes[1][i, 0], action[0] * 0.1, action[1] * 0.1, color='g')

    ax_top.scatter(*codebook_axes, c=colors)

    label = f'Var[{dim}]'
    label += f" | Total = {var[dim]}"
    norm = mpl.colors.Normalize(vmin=min_var, vmax=max_var)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax_bot, orientation='horizontal', label=label)
    cb.outline.set_visible(False)
    cb.set_ticks([])

    if not DO_3D:
        ax_top.set_aspect('equal')
    else:

        # manual
        plt_utils.equal_aspect_3d(ax_top, *codebook_axes, sgns=sgns)
        # First remove fill
        ax_top.view_init(elev=10., azim=90)
        ax_top.grid(False)
        ax_top.xaxis.pane.fill = False
        ax_top.yaxis.pane.fill = False
        ax_top.zaxis.pane.fill = False
        # plt.rcParams['grid.color'] = "0.93"

        # Now set color to white (or whatever is "invisible")
        ax_top.xaxis.pane.set_edgecolor('w')
        ax_top.yaxis.pane.set_edgecolor('w')
        ax_top.zaxis.pane.set_edgecolor('w')

    ax_top.set_xlabel("x")
    ax_top.set_ylabel("y")
    if DO_3D:
        ax_top.set_zlabel("z")

    ax_bot.set_aspect(0.05)

    ax_top.set_axis_off()

# xmin = min([ep[:, args.temporal_idx, ..., xa].min() for ep in to_save.values()])
# ymin = min([ep[:, args.temporal_idx, ..., ya].min() for ep in to_save.values()])
# xmax = max([ep[:, args.temporal_idx, ..., xa].max() for ep in to_save.values()])
# ymax = max([ep[:, args.temporal_idx, ..., ya].max() for ep in to_save.values()])
#
# if args.clip_x is not None:
#     xmin, xmax = args.clip_x
#
# if args.clip_y is not None:
#     ymin, ymax = args.clip_y
#
# logger.debug(f"X ({xmin}, {xmax}) - Y ({ymin}, {ymax}) --- clip_xy = ({args.clip_x is not None}, {args.clip_y is not None})")

# if args.scale_minmax:
#     ax.set_xlim([xmin, xmax][::sgns[0]])
#     ax.set_ylim([ymin, ymax][::sgns[1]])
# else:
#     all_max = np.array([xmax, ymax])
#     all_min = np.array([xmin, ymin])
#     max_range = (all_max - all_min).max()
#     center = 0.5 * (all_max + all_min)
#     e_max = center + max_range / 2
#     e_min = center - max_range / 2
#     ax.set_xlim([e_min[0], e_max[0]][::sgns[0]])
#     ax.set_ylim([e_min[1], e_max[1]][::sgns[1]])


# num_elements_per_key = np.product(num_elements_per_key)
# # num_elements_per_key = np.product(env_spec.names_to_shapes[args.key])
# assert num_elements_per_key >= 3, "There must be at least 3 axes to visualize (was %d)" % num_elements_per_key
# assert all([i != 0 for i in args.xyz_axes]), "Must be one indexed: %s" % args.xyz_axes
# xyz = [abs(i) - 1 for i in args.xyz_axes]
# xa, ya, za = xyz
# sgns = [int(i > 0) * 2 - 1 for i in args.xyz_axes]
# logger.debug("Using zero-indexed axes %s with scales %s" % (xyz, sgns))
#
# assert num_eps != 0, "No data to plot"
# assert all([args.ep_idxs[i] < num_eps for i in range(len(args.ep_idxs))]), "Bad idxs: %s" % args.ep_idxs
#
# episodes = sorted(list(set(args.ep_idxs)))
# if len(episodes) == 0:
#     episodes = list(range(num_eps))

if args.show:
    plt.show()
elif args.save_file:
    fig.savefig(args.save_file, transparent=True)
