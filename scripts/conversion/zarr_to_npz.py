import argparse

import zarr
import numpy as np
from attrdict import AttrDict as d

from muse.experiments import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('--episode_end_key', type=str, default='meta/episode_ends')
    parser.add_argument('--save_file', type=str, default=None)
    args = parser.parse_args()

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

    logger.debug('Parsed structure...')
    data = out['data']
    data.leaf_shapes().pprint()

    episode_ends = out[args.episode_end_key]

    data.done = np.zeros(data.get_one().shape[0], dtype=np.bool)
    data.done[episode_ends - 1] = True
    assert data.done[-1]

    if args.save_file is not None:
        logger.debug(f'Saving to --> {args.save_file}')
        np.savez_compressed(args.save_file, **data.as_dict())

    logger.debug('Done.')