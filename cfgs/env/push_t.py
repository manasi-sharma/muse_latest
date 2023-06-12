from attrdict import AttrDict as d

from muse.envs.pymunk.push_t import PushTEnv

export = PushTEnv.default_params & d(
    cls=PushTEnv,
)
