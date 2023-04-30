"""
Launch the evaluation sweep for square for bc-MLP
"""
import subprocess

from configs.utils import hr_name

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--noise_type', type=str, default='s', choices=['s', 'p', 'h'])
parser.add_argument('--num_rollouts', type=int, default=100)
parser.add_argument('--model_file', type=str, default='best_model.pt')
parser.add_argument('--extra_env_args', type=str, default='')
parser.add_argument('--dset_suffs', type=str, nargs='+', default=['', '-first100', '-first50'])
parser.add_argument('--dset_override', type=str, default=None)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()


def sweep_from_arr(arr):
    s = "[[[ "
    for a in arr:
        s = s + str(a) + " "
    s = s + "]]]"
    return s

num_rollouts = args.num_rollouts
noise_type = args.noise_type

s_noises = [0.05, 0.1, 0.2]
typed_noises = {
    'h': [0],  # human has no noise
    'p': [0, 0.005, 0.01, 0.02],
    's': s_noises,
}[noise_type]

dset_suffs = list(args.dset_suffs)

model_file = args.model_file
if model_file == 'best_model.pt':
    model_prefix = 'best_'
elif model_file == 'model.pt':
    model_prefix = ''
else:
    # otherwise remove extension
    mode_prefix = model_file[:-3] + "_"

if args.dset_override is not None:
    dset_suffs = [args.dset_override]

batch_sizes = [256 for _ in dset_suffs]  # 50 if suff == '-first10' else
# batch_size = 256

for batch_size, suff in zip(batch_sizes, dset_suffs):
    for n in typed_noises:
        hr_s_sweep = sweep_from_arr(s_noises)
        # special case for zero noise
        if n == 0:
            noise = ""
        else:
            noise = f"_l{noise_type}n{hr_name(n)}"

        save_names = [f"{model_prefix}eval{num_rollouts}_lsn{hr_name(inner_s)}.npz" for inner_s in s_noises]
        hr_names = sweep_from_arr(save_names)

        # single dataset overrides the list of suffixes.
        if args.dset_override is not None:
            dset = suff
        else:
            assert noise_type != 'h', "Human dataset requires dset override!"
            dset = f"scripted_rm{noise}_square_200ep{suff}"

        exp_name = f'experiments/hvs/velact_b{batch_size}_h10_{dset}_bc-l2_mlp400-d2'
        # command to run
        command = f"python scripts/slurm.py -sp -c iliad --cpus {2 * len(s_noises)} --job-name sqeval{noise}{suff} --- " \
                  f"python scripts/collect.py --max_eps {num_rollouts} --save_file {hr_names} " \
                  f"--save_every_n_episodes 100 --model_file {model_file} --track_returns exp={exp_name} " \
                  f"%env_train --pos_noise_std {hr_s_sweep} {args.extra_env_args}"

        print(command)
        if not args.test:
            subprocess.run(command, shell=True)
        print('----------------------------------------------------')
