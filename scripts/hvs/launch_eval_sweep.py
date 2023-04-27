"""
Launch the evaluation sweep
"""
import subprocess

from configs.utils import hr_name

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--noise_type', type=str, default='s', choices=['s', 'p'])
parser.add_argument('--num_rollouts', type=int, default=100)
parser.add_argument('--model_file', type=str, default='best_model.pt')
parser.add_argument('--extra_env_args', type=str, default='')
args = parser.parse_args()


def sweep_from_arr(arr):
    s = "[[[ "
    for a in arr:
        s = s + str(a) + " "
    s = s + "]]]"
    return s

num_rollouts = args.num_rollouts
noise_type = args.noise_type

s_noises = [0.01, 0.02, 0.03, 0.04, 0.05]
dset_suffs = ['', '-first100', '-first50']

model_file = args.model_file
if model_file == 'best_model.pt':
    model_prefix = 'best_'
elif model_file == 'model.pt':
    model_prefix = ''
else:
    # otherwise remove extension
    mode_prefix = model_file[:-3] + "_"

for suff in dset_suffs:
    for s in s_noises:
        hr_s_sweep = sweep_from_arr(s_noises)
        noise = f"{noise_type}n{hr_name(s)}"

        save_names = [f"{model_prefix}eval{num_rollouts}_{noise_type}n{hr_name(inner_s)}.npz" for inner_s in s_noises]
        hr_names = sweep_from_arr(save_names)

        # dset = f"pm_direct_{noise}_1000ep{suff}"
        dset = f"all_pm_direct_{noise}_ns500_1000ep{suff}"

        exp_name = f'experiments/pmhvs/velact_b256_h10_{dset}_bc-l2_mlp200-d2'
        # command to run
        command = f"python scripts/slurm.py -sp -c iliad --cpus 10 --job-name pmeval_{noise}{suff} --- " \
                  f"python scripts/collect.py --max_eps {num_rollouts} --save_file {hr_names} " \
                  f"--save_every_n_episodes 100 --model_file {model_file} --track_returns exp={exp_name} " \
                  f"%env_train --noise_std {hr_s_sweep} {args.extra_env_args}"

        print(command)
        subprocess.run(command, shell=True)
        print('----------------------------------------------------')
