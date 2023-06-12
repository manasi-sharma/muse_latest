from attrdict import AttrDict as d

from cfgs.env import slider
from configs.fields import GroupField
from muse.envs.pymunk.slider_env_2d import SliderBlockEnv2D
from muse.models.model import Model
from muse.policies.pymunk.pymunk_policies import PymunkDirectPolicy
export = d(
    exp_name='sliderhvs/collection',
    env_spec=GroupField('env_train', SliderBlockEnv2D.get_default_env_spec_params),
    env_train=slider.export,
    model=d(cls=Model, ignore_inputs=True),
    policy=d(
        cls=PymunkDirectPolicy,
        speed=1.0,  # max speed
        noise_std=0.,
    ),
)
