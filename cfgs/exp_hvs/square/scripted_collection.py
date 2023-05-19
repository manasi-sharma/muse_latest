from attrdict import AttrDict as d

from cfgs.env import square
from configs.fields import GroupField, Field as F
from muse.envs.robosuite.robosuite_env import RobosuiteEnv
from muse.models.model import Model
from muse.policies.meta_policy import MetaPolicy
from muse.policies.scripted.robosuite_policies import WaypointPolicy, get_nut_assembly_square_policy_params


def get_next_param_fn(**kwargs):
    return lambda idx, _, obs, goal, env=None, **inner_kwargs: \
        (0, get_nut_assembly_square_policy_params(obs, goal, env=env, **kwargs))


export = d(
    exp_name='hvs/collection',

    env_spec=GroupField('env_train', RobosuiteEnv.get_default_env_spec_params),
    env_train=square.export,

    model=d(cls=Model, ignore_inputs=True),

    policy=d(
        cls=MetaPolicy,
        num_policies=1,
        max_policies_per_reset=1,  # one policy then terminate
        policy_0=d(cls=WaypointPolicy),
        random_motion=False,
        lin_vel_std=0.,
        ang_vel_std=0.,
        next_param_fn=F(['random_motion', 'lin_vel_std', 'ang_vel_std'],
                        lambda *v: get_next_param_fn(random_motion=v[0], lin_vel_noise=v[1], ang_vel_noise=v[2])),
    ),
)
