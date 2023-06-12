from attrdict import AttrDict as d

from cfgs.env import push_t
from configs.fields import GroupField
from muse.envs.pymunk.push_t import PushTEnv
from muse.models.model import Model
from muse.policies.mouse_teleop_policy import MouseTeleopPolicy

export = d(
    exp_name='push_t/teleop',
    cls=PushTEnv,
    env_spec=GroupField('env_train', PushTEnv.get_default_env_spec_params),
    env_train=push_t.export & d(render=True),

    policy=d(
        cls=MouseTeleopPolicy
    ),
    model=d(
        cls=Model,
        ignore_inputs=True
    )
)
