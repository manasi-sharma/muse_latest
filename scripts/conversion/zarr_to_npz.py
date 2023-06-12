import argparse

import zarr
import numpy as np
from attrdict import AttrDict as d

from muse.experiments import logger

env_to_parser_map = {
    'push_t': {
        'state': [('agent/position', 2), ('block/position', 2), ('block/angle', 1)],
    },
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('--episode_end_key', type=str, default='meta/episode_ends')
    parser.add_argument('--remap_img_key', type=str, default='img')
    parser.add_argument('--save_file', type=str, default=None)
    parser.add_argument('--env_name', type=str, default=None, choices=list(env_to_parser_map.keys()),
                        help='if specified, will do parsing to add new keys based on the env')
    args = parser.parse_args()

    names_to_new_names = None
    if args.env_name is not None:
        names_to_new_names = env_to_parser_map[args.env_name]

    root = zarr.open(args.file)

    logger.debug('Original structure...')
    print(root.tree())

    out = d()

    def copy_in(r, prefix=''):
        for k, v in r.arrays():
            out[prefix + k] = v[:]
        for k, v in r.groups():
            copy_in(v, prefix=prefix + k + "/")

    copy_in(root)

    data = out['data']

    if names_to_new_names is not None:
        for name, new_names in names_to_new_names.items():
            new_names, new_shapes = [n[0] for n in new_names], [n[1] for n in new_names]
            val = data[name]
            each_arr = np.split(val, np.cumsum(new_shapes)[:-1], axis=-1)
            for key, arr in zip(new_names, each_arr):
                data[key] = arr

    if args.remap_img_key is not None:
        data['image'] = data[args.remap_img_key]
        del data[args.remap_img_key]

    logger.debug('Parsed structure...')
    data.leaf_shapes().pprint()

    episode_ends = out[args.episode_end_key]

    data.done = np.zeros(data.get_one().shape[0], dtype=bool)
    data.done[episode_ends - 1] = True
    assert data.done[-1]

    if args.save_file is not None:
        logger.debug(f'Saving to --> {args.save_file}')
        np.savez_compressed(args.save_file, **data.as_dict())

    logger.debug('Done.')