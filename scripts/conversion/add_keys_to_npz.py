"""
Copy (keys) from (keys_file) to (file)
"""
import argparse

import numpy as np
from attrdict import AttrDict as d

from muse.experiments import logger
from muse.utils.input_utils import query_string_from_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='NPZ file to load everything from')
    parser.add_argument('--keys_file', type=str, required=True, help='NPZ file to load keys from')
    parser.add_argument('--keys', type=str, nargs='+', required=True, help='Keys to load from keys_file')
    parser.add_argument('--keys_out', type=str, nargs='*', default=None, help='Keys to write in load_fils')
    parser.add_argument('--output_file', type=str, required=True, help='NPZ file to save to')
    parser.add_argument('--no_overwrite', action='store_true', help='No overlap between keys and whats in file')
    parser.add_argument('--y', action='store_true', help='save without checking')
    args = parser.parse_args()

    logger.debug(f"Loading {args.file}...")
    dc = d.from_dict(dict(np.load(args.file, allow_pickle=True)))
    logger.debug(f"Keys: {dc.list_leaf_keys()}")

    logger.debug(f"Loading {args.keys_file}")
    dc_with_keys = np.load(args.keys_file, allow_pickle=True)
    logger.debug(f"Keys: {list(dc_with_keys.keys())}")

    if args.keys_out is None:
        args.keys_out = args.keys
    else:
        assert len(args.keys_out) == len(args.keys), "Must provide same number of out keys as in keys!"

    for key, key_out in zip(args.keys, args.keys_out):
        assert not args.no_overwrite or key_out not in dc.leaf_keys(), f"{key_out} is in file, but no_overwrite=True!"
        if key_out in dc.leaf_keys():
            logger.warn(f"Overwriting {key_out} in dc!")
        dc[key_out] = dc_with_keys[key]

    logger.debug(f"New shapes:")
    dc.leaf_shapes().pprint()

    do_save = args.y or query_string_from_set(f'Save to {args.output_file}? (y/n)', ['y', 'n']) == 'y'
    if do_save:
        logger.warn('Saving...')
        np.savez_compressed(args.output_file, **dc.as_dict())

    logger.debug("Done.")
