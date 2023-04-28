from configs.fields import Field
from muse.envs.bullet_envs.block3d.block_env_3d import BlockEnv3D
from muse.envs.bullet_envs.block3d.reward_fns import get_lift_reward

from attrdict import AttrDict as d

reward_map = {
    'lift': get_lift_reward(lift_distance=0.15)
}

export = BlockEnv3D.default_params & d(
    cls=BlockEnv3D,
    action_mode='ee_euler_delta',
    num_blocks=1,
    do_random_ee_position=False,
    img_width=128,
    img_height=128,
    render=False,
    compute_images=True,
    task_name=None,
    env_reward_fn=Field('task_name', lambda name: reward_map[name] if name is not None else None),
    done_reward_thresh=1.,
)
