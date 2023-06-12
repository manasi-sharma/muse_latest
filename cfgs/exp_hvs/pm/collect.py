from attrdict import AttrDict as d

from cfgs.env import fixed_pm
from configs.fields import GroupField
from muse.envs.simple.point_mass_env import PointMassEnv
from muse.models.model import Model
from muse.policies.pm.pm_policies import PMDirectPolicy
export = d(
    exp_name='pmhvs/collection',
    env_spec=GroupField('env_train', PointMassEnv.get_default_env_spec_params),
    env_train=fixed_pm.export,
    model=d(cls=Model, ignore_inputs=True),
    policy=d(
        cls=PMDirectPolicy,
        speed=1.0,  # max speed
        noise_std=0.,
    ),
)
