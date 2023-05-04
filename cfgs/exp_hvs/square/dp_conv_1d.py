from attrdict import AttrDict as d

from cfgs.exp_hvs.square import bc_rnn
from cfgs.model import dp_conv_1d
from cfgs.trainer import ema
from configs.fields import Field as F
from muse.utils.general_utils import dc_skip_keys

export = dc_skip_keys(bc_rnn.export, 'model') & d(
    use_ema=False,
    horizon=16,
    exp_name='hvsBlock3D/velact_{?seed:s{seed}_}b{batch_size}_h{horizon}{?use_ema:_ema}_{dataset}',
    model=dp_conv_1d.export & d(
        device=F('device'),
        action_decoder=d(
            horizon=F('horizon'),
        )
    ),
    trainer=d(
        use_stabilizer=F('use_ema'),
        stabilizer=ema.export,
    )
)
