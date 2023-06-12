from attrdict import AttrDict as d

from cfgs.exp_hvs.pusht import bc_rnn
from cfgs.model import dp_conv_1d
from cfgs.trainer import ema
from configs.fields import Field as F
from muse.utils.general_utils import dc_skip_keys

export = dc_skip_keys(bc_rnn.export, 'model') & d(
    use_ema=False,
    horizon=16,
    exp_name='push_t/posact{?use_keypoint:-kp}_{?seed:s{seed}_}b{batch_size}_h{horizon}{?use_ema:_ema}_{dataset}',
    model=dp_conv_1d.export & d(
        state_names=F('use_keypoint', lambda kp: ['state', 'keypoint'] if kp else ['state']),
        device=F('device'),
        action_decoder=d(
            use_tanh_out=False,
            horizon=F('horizon'),
        )
    ),
    trainer=d(
        use_stabilizer=F('use_ema'),
        stabilizer=ema.export,
    )
)
