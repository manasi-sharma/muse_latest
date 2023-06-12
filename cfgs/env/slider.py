from attrdict import AttrDict as d

from muse.envs.pymunk.slider_env_2d import SliderBlockEnv2D

export = SliderBlockEnv2D.default_params & d(
    cls=SliderBlockEnv2D,
)