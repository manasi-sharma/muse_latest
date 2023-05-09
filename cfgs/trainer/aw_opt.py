import torch
from attrdict import AttrDict as d

from muse.trainers.optimizers.optimizer import SingleOptimizer

export = d(
    cls=SingleOptimizer,
    max_grad_norm=None,
    base_optimizer=d(
        cls=torch.optim.AdamW,
        lr=1.0e-4,
        betas=[0.95, 0.999],
        eps=1.0e-8,
        weight_decay=1.0e-6,
    ),
)
