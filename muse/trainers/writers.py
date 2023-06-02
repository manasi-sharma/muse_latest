from typing import List

from attrdict.utils import get_with_default

from muse.experiments import logger
from muse.experiments.file_manager import ExperimentFileManager


class Writer(object):
    """
    Writers are abstractions for logging experiment information

    TODO
    """

    def __init__(self, exp_name: str, params: object, file_manager: object,
                 resume: bool = False) -> object:
        self.exp_name = exp_name
        self.params = params.leaf_copy()
        self._file_manager = file_manager
        self.resume = resume

        self._init_params_to_attrs(params)

    def _init_params_to_attrs(self, params):
        self.project_name = get_with_default(params, "project_name", 'muse')
        self.config = dict(get_with_default(params, "config", {}))

    def open(self):
        """
        Opens a channel for whatever type of writer this is

        Returns
        -------

        """
        raise NotImplementedError

    def update_config(self, cfg_dict):
        """
        Adds a config to the logger after the open() call.
        Parameters
        ----------
        cfg_dict: dict

        Returns
        -------

        """
        raise NotImplementedError

    def add_scalar(self, name, value, step, **kwargs):
        """
        Logs a scalar.

        Parameters
        ----------
        name
        value
        step
        kwargs

        Returns
        -------
        """
        raise NotImplementedError


class TensorboardWriter(Writer):

    def open(self):
        from torch.utils.tensorboard import SummaryWriter
        self._summary_writer = SummaryWriter(self._file_manager.exp_dir)

    def update_config(self, cfg_dict):
        # tensorboard does not have a config option I think
        pass

    def add_scalar(self, name, value, step, **kwargs):
        self._summary_writer.add_scalar(name, value, step, **kwargs)


class WandbWriter(Writer):

    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)
        self.tags = params << "tags"
        self.force_id = params << "force_id"
        if self.tags:
            assert isinstance(self.tags, List), f"Tags {self.tags} must be a list!"
            logger.debug(f'[wandb] Using tags: {self.tags}')
        if self.resume and self.force_id is None:
            import wandb
            # set the force id bc resuming with wandb is stupid
            api = wandb.Api()
            all_runs = api.runs(self.project_name)
            found = False
            # searching for the right run, not sure how to do this with query
            for r in all_runs:
                if r.name == self.exp_name:
                    self.force_id = r.id
                    found = True
            if not found:
                logger.warn('Could not find a run id but resume=True!')

    def open(self):
        import wandb
        import os
        os.environ['WANDB__SERVICE_WAIT'] = "300"
        self.run = wandb.init(project=self.project_name, name=self.exp_name,
                              config=self.config, resume=self.resume, tags=self.tags, id=self.force_id)

    def update_config(self, cfg_dict):
        self.run.config.update(cfg_dict)

    def add_scalar(self, name, value, step, **kwargs):
        self.run.log({name: value}, step=step)
