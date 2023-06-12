from abc import ABC

import torch.nn as nn
from typing import List

from muse.datasets.dataset import Dataset
from muse.experiments import logger
from muse.models.model import Model
from attrdict import AttrDict as d
from attrdict.utils import get_with_default

from muse.utils.general_utils import params_to_object


class Optimizer(object):
    """
    Optimizers define a step() call that updates a model given a loss.
    This encompasses schedulers as well (see SingleOptimizer)

    See GoalTrainer for an example of how to use one.
    """
    def __init__(self, params: d, model: Model, datasets: List[Dataset] = None):
        self._model = model
        self._params = params.leaf_copy()
        self._datasets = datasets if datasets is not None else []

        self._init_params_to_attrs(params)
        self._init_setup()

    def _init_params_to_attrs(self, params: d):
        self._max_grad_norm = get_with_default(params, "max_grad_norm", None)

    def _clip_grad_norm(self, model=None):
        model = model if model is not None else self._model
        if self._max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), self._max_grad_norm)  # max

    def _init_setup(self):
        pass

    def step(self, loss, inputs, outputs, dataset_idx, meta: d = d(), i=0, ti=0, writer=None, writer_prefix="", **kwargs):
        raise NotImplementedError

    @property
    def param_groups(self):
        raise NotImplementedError

    def get_optim_parameters(self, **kwargs):
        """ Get the parameters for the optimizer (kwargs will contain info to distinguish which optimizer)

        Parameters
        ----------
        kwargs

        Returns
        -------
        Iterable of parameters, or a dictionary of named parameters

        """
        return self._model.parameters()


class SingleOptimizer(Optimizer):
    """
    The default single optimizer, which steps the model and defines a default scheduler

    params:
        base_optimizer: AttrDict that with cls(torch.Optimizer), which fills in model.parameters()
        base_scheduler: [optional] AttrDict with cls(torch Scheduler), a param will be filled in for optimizer=<>

    """

    def _init_params_to_attrs(self, params: d):
        super(SingleOptimizer, self)._init_params_to_attrs(params)
        self._base_optimizer_params = params["base_optimizer"]
        if "base_scheduler" in params.leaf_keys():
            logger.debug("SingleOptimizer using scheduler..")
            self._base_scheduler_params = params['base_scheduler']
        else:
            self._base_scheduler_params = None

    def _init_setup(self):
        self._base_optimizer = params_to_object(self._base_optimizer_params, self.get_optim_parameters())
        if self._base_scheduler_params is not None:
            self._base_scheduler = params_to_object(self._base_scheduler_params)
        else:
            self._base_scheduler = None

    def step(self, loss, inputs, outputs, dataset_idx, meta: d = d(), i=0, ti=0,
             writer=None, writer_prefix="", **kwargs):
        # loss might be a dictionary potentially. TODO
        self._base_optimizer.zero_grad()
        loss.backward()
        if self._max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)  # max

        # grad norm writing, only when writer is provided. TODO make this optional
        if writer is not None:
            grad_norms = 0.
            for p in self._model.parameters():
                if p.grad is not None:
                    grad_norms += p.grad.data.norm(2).pow(2).item()
            writer.add_scalar('train/policy_grad_norms', grad_norms, i)

        self._base_optimizer.step()
        self._scheduler_step(loss, inputs, outputs, dataset_idx, meta=meta)

    def _scheduler_step(self, loss, inputs, outputs, dataset_idx, meta: d = d()):
        if self._base_scheduler is not None:
            self._base_scheduler.step()

    @property
    def param_groups(self):
        return self._base_optimizer.param_groups


class MultiOptimizer(Optimizer, ABC):
    """
    Defines multiple optimizers (abstract).

    params:
        num_optimizers: int
        get_optimizer: Callable that returns the torch.Optimizer given model.parameters() and an index
        get_scheduler: [optional] Callable that returns the torch Scheduler given optimizer[index] and the index
        loss_names: [optional] List[str] a list of the loss name for each optimizer.

    Child class should define step() to use the above.
    """

    def _init_params_to_attrs(self, params: d):
        super(MultiOptimizer, self)._init_params_to_attrs(params)
        self._num_optimizers = params["num_optimizers"]
        self._get_optimizer = params["get_optimizer"]
        # if None, all optimizers use the same loss, which gets passed in.
        self._loss_names = get_with_default(params, "loss_names", None)

        # loss names are used to parse the loss dict (input).
        assert self._loss_names is None or len(self._loss_names) == self._num_optimizers, [self._loss_names, self._num_optimizers]

        if "get_scheduler" in params.leaf_keys():
            logger.debug("MultiOptimizer using scheduler..")
            self._get_scheduler = params.get_scheduler
        else:
            self._get_scheduler = None

    def _init_setup(self):
        self._optimizers = []
        self._schedulers = []
        for i in range(self._num_optimizers):
            base_opt = self._get_optimizer(self._model, i)  # let the function figure this out.
            base_sched = self._get_scheduler(base_opt, i) if self._get_scheduler is not None else None  # let the function figure this out.
            self._optimizers.append(base_opt)  # None is allowed.
            self._schedulers.append(base_sched)  # None is allowed.

    @property
    def param_groups(self):
        all_groups = []
        for optim in self._optimizers:
            all_groups.extend(optim.param_groups)
        return all_groups
