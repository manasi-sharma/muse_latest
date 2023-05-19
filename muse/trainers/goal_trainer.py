import math
import numpy as np
import torch
from typing import Callable

from muse.datasets.preprocess.data_augmentation import DataAugmentation
from muse.experiments import logger
from muse.trainers.base_goal_trainer import BaseGoalTrainer
from muse.trainers.optimizers.optimizer import SingleOptimizer
from muse.trainers.stabilizers import Stabilizer
from muse.utils.general_utils import is_next_cycle, listify, timeit
from attrdict import AttrDict
from attrdict.utils import get_with_default, get_cls_param_instance


class GoalTrainer(BaseGoalTrainer):

    def __init__(self, params, file_manager,
                 model,
                 policy,
                 goal_policy,
                 datasets_train,
                 datasets_holdout,
                 env_train,
                 env_holdout,
                 policy_holdout=None,
                 goal_policy_holdout=None,
                 reward=None,
                 writer=None,
                 optimizer=None,
                 sampler=None, ):
        """
        Trainer for a goal_policy, if you want online env steps during training.
        NOTE: env will be reset when either env is terminated or goal_policy is terminated.
        TODO option for terminate when policy is done

        Parameters
        ----------
        params: parameters for training
        file_manager: manages
        model: a global model, with all the parameters necessary for computation at each level.
        policy: produces actions for train env
        goal_policy: produces goals for train policy/eng
        datasets_train: a list of training datasets to maintain
        datasets_holdout: a list of holdout datasets to maintain.
        env_train: the environment to step during training.
        env_holdout: the environment to step during holdout.
        policy_holdout: produces actions for the holdout env, None means same as train.
        goal_policy_holdout: produces goals for the holdout policy/env, None means same as train.
        reward: A Reward object to compute rewards, None means use the environment reward.
        writer: A Writer object (or None, in which case we will create a Tensorboard writer)
        optimizer: Optimizer object to step the model.
        sampler: Sampler or sampler params AttrDict(cls, ...) to use with the datasets.
        """
        self._datasets_train = datasets_train
        self._datasets_holdout = datasets_holdout

        logger.debug(f'Using {len(self._datasets_train)} training datasets, '
                     f'and {len(self._datasets_holdout)} holdout datasets')

        # get samplers
        self._dataset_samplers_train = self._init_samplers(datasets_train, sampler, group_name='train')
        self._dataset_samplers_holdout = self._init_samplers(datasets_holdout, sampler, group_name='holdout')

        super(GoalTrainer, self).__init__(params, file_manager, model, policy, goal_policy, env_train, env_holdout,
                                          policy_holdout, goal_policy_holdout, reward=reward, optimizer=optimizer,
                                          writer=writer)

    @staticmethod
    def _init_samplers(datasets, sampler, group_name='train'):
        """ Initializing sampler for a group of datasets.

        Either instantiate with given sampler class using all the datasets (if sampler specified)
        or instantiate individually.

        Parameters
        ----------
        datasets: List[Dataset]
        sampler: Sampler
        group_name: str

        Returns
        -------

        """
        samplers = []
        for i, ds in enumerate(datasets):
            if sampler is not None:
                # first dataset will be the current one by default.
                samplers.append(sampler.cls([ds] + datasets[:i] + datasets[i + 1:], sampler))
            else:
                samplers.append(ds.get_sampler())
        return samplers

    def _init_params_to_attrs(self, params):
        super(GoalTrainer, self)._init_params_to_attrs(params)

        # BASE STEPS
        self._max_steps = int(params.max_steps)

        # DATA AUGMENTATION
        self.data_augmentation_params: AttrDict = get_with_default(params, "data_augmentation_params", AttrDict())
        if isinstance(self.data_augmentation_params, DataAugmentation):
            self.data_augmentation: DataAugmentation = self.data_augmentation_params  # allow passing in data aug
        elif not self.data_augmentation_params.is_empty():
            self.data_augmentation: DataAugmentation = get_cls_param_instance(self.data_augmentation_params, "cls",
                                                                              None, DataAugmentation)
            if len(self._datasets_train) == 1:
                self.data_augmentation.link_dataset(self._datasets_train[0])
        else:
            logger.info("Using no data augmentation.")
            self.data_augmentation = None
        self._train_do_data_augmentation = get_with_default(params, "train_do_data_augmentation", True)

        # TRAINING STATE
        self._current_train_step = [0] * len(self._datasets_train)
        self._current_train_loss = [math.inf] * len(self._datasets_train)
        self._current_holdout_step = [0] * len(self._datasets_holdout)
        self._current_holdout_loss = [math.inf] * len(self._datasets_holdout)
        # each dataset gets their own train/holdout every n steps
        self._train_every_n_steps = listify(params["train_every_n_steps"], len(self._datasets_train))
        self._holdout_every_n_steps = listify(params["holdout_every_n_steps"], len(self._datasets_holdout))

        # env STEP (one step)
        self._step_train_env_every_n_steps = int(params["step_train_env_every_n_steps"])
        self._step_train_env_n_per_step = get_with_default(params, "step_train_env_n_per_step", 1)
        self._step_holdout_env_every_n_steps = int(params["step_holdout_env_every_n_steps"])
        self._step_holdout_env_n_per_step = get_with_default(params, "step_holdout_env_n_per_step", 1)
        self._log_every_n_steps = int(params["log_every_n_steps"])
        self._save_every_n_steps = int(params["save_every_n_steps"])

        # env ROLLOUT (step until done)
        self._rollout_train_env_every_n_steps = get_with_default(params, "rollout_train_env_every_n_steps", 0)
        self._rollout_train_env_n_per_step = get_with_default(params, "rollout_train_env_n_per_step", 1)
        self._rollout_holdout_env_every_n_steps = get_with_default(params, "rollout_holdout_env_every_n_steps", 0)
        self._rollout_holdout_env_n_per_step = get_with_default(params, "rollout_holdout_env_n_per_step", 1)

        assert self._rollout_holdout_env_every_n_steps == 0 or self._step_train_env_every_n_steps == 0, \
            "Cannot both rollout AND step train env."
        assert self._rollout_holdout_env_every_n_steps == 0 or self._step_holdout_env_every_n_steps == 0, \
            "Cannot both rollout AND step holdout env."

        # add to the training dataset every N completed goals, or if episode terminates.
        #   This helps if resets are infrequent, or if we do not care about the episode boundary.
        self._add_to_data_train_every_n_goals = params["add_to_data_train_every_n_goals"]
        self._add_to_data_holdout_every_n_goals = params["add_to_data_holdout_every_n_goals"]

        # data train save freq, per dataset.
        self._save_data_train_every_n_steps = listify(params["save_data_train_every_n_steps"],
                                                      len(self._datasets_train))
        # data holdout save freq, per dataset.
        self._save_data_holdout_every_n_steps = listify(params["save_data_holdout_every_n_steps"],
                                                        len(self._datasets_holdout))
        self._block_train_on_first_n_steps = int(get_with_default(params, "block_train_on_first_n_steps", 0))
        self._block_env_on_first_n_steps = int(get_with_default(params, "block_env_on_first_n_steps", 0))
        self._write_to_tensorboard_every_n = int(
            get_with_default(params, "write_to_tensorboard_every_n_train_steps", 20))

        # stabilizer
        self._use_stabilizer = get_with_default(params, "use_stabilizer", False)

        self._save_checkpoint_every_n_steps = get_with_default(params, "save_checkpoint_every_n_steps", 0)
        if self._save_checkpoint_every_n_steps > 0:
            assert self._save_checkpoint_every_n_steps % self._save_every_n_steps == 0, "Checkpointing steps should be a multiple of model save steps."

    def _init_optimizers(self, params):
        """
            override this to make different optimizer schemes
            NOTE: if this is overridden, need to override train_step as well

        :param params:
        :return:
        """
        if len(list(self._model.parameters())) > 0:
            if self._optimizer is None:
                self._optimizer = SingleOptimizer(params["optimizer"], self._model, datasets=self._datasets_train)
            elif isinstance(self._optimizer, AttrDict):
                self._optimizer = self._optimizer.cls(self._optimizer, self._model,
                                                      datasets=self._datasets_train)
        else:
            logger.warn("Model has no parameters...")

    def _optimizer_step(self, loss, inputs, outputs, dataset_idx, meta: AttrDict = AttrDict(), i=0, ti=0, writer=None,
                        writer_prefix=""):
        # loss might be a dictionary potentially. TODO
        return self._optimizer.step(loss, inputs, outputs, dataset_idx, meta=meta, i=i, ti=ti, writer=writer,
                                    writer_prefix=writer_prefix)

    def _init_stabilizers(self, params):
        self._stb = None
        if self._use_stabilizer:
            logger.debug(f'Using stabilizer of class {params.stabilizer.cls}')
            self._stb = params.stabilizer.cls(self._model,
                                              **params.stabilizer.node_leaf_without_keys(['cls']).as_dict())
            assert isinstance(self._stb, Stabilizer), "instantiated object is not a Stabilizer instance!"

    def _stabilizer_step(self, loss, inputs, outputs, dataset_idx, meta: AttrDict = AttrDict(), i=0, ti=0,
                         writer=None,
                         writer_prefix=""):
        if self._stb is not None:
            self._stb.step(self._model, ti=ti)
            # update the model used for eval / saving
            self._eval_model = self._stb.stable_model

    def _write_step(self, model, loss, inputs, outputs, meta: AttrDict = AttrDict(), dataset_idx=0, **kwargs):
        self._summary_writer.add_scalar("train_step", self._current_train_step[dataset_idx], self._current_step)
        super(GoalTrainer, self)._write_step(model, loss, inputs, outputs, meta)

    def _train_step(self, model, dataset_idx):
        """ Trains a model for a single iteration using the given dataset idx to select a dataset to sample from.

        (1) gets a batch
        (2) does any augmentation
        (3) call loss + optimizer.step or call train_step
        (4) write any additional values.

        Parameters
        ----------
        model
        dataset_idx

        Returns
        -------

        """
        if len(self._datasets_train[dataset_idx]) == 0:
            logger.warn("Skipping training step since dataset is empty.")
            return

        # (B x H x ...)
        with timeit('train/get_batch'):
            with torch.no_grad():
                sampler = self._dataset_samplers_train[dataset_idx]
                indices = sampler.get_indices()
                res = self._datasets_train[dataset_idx].get_batch(indices=indices,
                                                                  torch_device=model.device)
                inputs, outputs = res[:2]
                meta = res[2] if len(res) == 3 else AttrDict()

        if self._train_do_data_augmentation and self.data_augmentation is not None:
            with timeit('train/data_augmentation'):
                inputs, outputs = self.data_augmentation.forward(inputs, outputs)

        model.train()

        sw = None
        if is_next_cycle(self._current_train_step[dataset_idx], self._write_to_tensorboard_every_n):
            sw = self._summary_writer

        if model.implements_train_step:
            # Model defines train_step
            with timeit('train/model_train_step'):
                loss = model.train_step(inputs, outputs, i=self._current_step,
                                        ti=self._current_train_step[dataset_idx], writer=sw,
                                        writer_prefix="train/", training=True, meta=meta,
                                        optimizer=self._optimizer,
                                        dataset_idx=dataset_idx,
                                        dataset=self._datasets_train[dataset_idx])
        else:
            # default train step computes loss and optimizes it.
            with timeit('train/loss'):
                loss = model.loss(inputs, outputs, i=self._current_step, writer=sw,
                                  writer_prefix="train/", training=True, meta=meta, dataset_idx=dataset_idx,
                                  dataset=self._datasets_train[dataset_idx])

            with timeit('train/backprop'):
                self._optimizer_step(loss, inputs, outputs, dataset_idx, meta=meta, i=self._current_step,
                                     ti=self._current_train_step[dataset_idx], writer=sw, writer_prefix="train/")

            with timeit('train/stabilizer'):
                self._stabilizer_step(loss, inputs, outputs, dataset_idx, meta=meta, i=self._current_step,
                                      ti=self._current_train_step[dataset_idx], writer=sw, writer_prefix="train/")

        with timeit('train/detach_loss'):
            if isinstance(loss, AttrDict):
                loss = loss.loss
                if isinstance(loss, Callable):
                    loss = loss()
            self._current_train_loss = loss.item()

        if sw is not None:
            with timeit("writer"):
                self._write_step(model, loss, inputs, outputs, meta=meta, dataset_idx=dataset_idx)

    def _holdout_step(self, model, dataset_idx):
        # (B x H x ...)
        with timeit('total_holdout_step'):
            sampler = self._dataset_samplers_holdout[dataset_idx]
            res = self._datasets_holdout[dataset_idx].get_batch(indices=sampler.get_indices(),
                                                                torch_device=model.device)
            inputs, outputs = res[:2]
            meta = res[2] if len(res) == 3 else AttrDict()
            model.eval()
            with torch.no_grad():
                loss = model.loss(inputs, outputs, i=self._current_step, writer=self._summary_writer,
                                  writer_prefix="holdout/", training=False, meta=meta, dataset_idx=dataset_idx,
                                  dataset=self._datasets_holdout[dataset_idx])
                if isinstance(loss, AttrDict):
                    loss = loss.loss  # just get one.
                self._current_holdout_loss = loss.item()

        with timeit("writer"):
            if self._summary_writer is not None:
                self._summary_writer.add_scalar("holdout_step", self._current_holdout_step[dataset_idx],
                                                self._current_step)

    def _get_save_meta_data(self):
        return {
            'train_step': self._current_train_step,
            'holdout_step': self._current_holdout_step,
            'train_loss': self._current_train_loss,
            'holdout_loss': self._current_holdout_loss
        }

    def _restore_meta_data(self, checkpoint):
        self._current_train_step = list(checkpoint['train_step'])  # todo add env
        self._current_holdout_step = list(checkpoint['holdout_step'])
        self._current_train_loss = checkpoint['train_loss']
        self._current_holdout_loss = checkpoint['holdout_loss']

    def _log(self):  # , AVG RETURN: {}±{}' TODO
        logger.info(
            f'[{self._current_step}] env (steps={self._current_env_train_step}, eps={self._current_env_train_ep}) '
            f'| env holdout (steps={self._current_env_holdout_step}, eps={self._current_env_holdout_ep})'
            f'\n\t\t\t\tdataset: (steps, loss):' + \
            f'\n\t\t\t\tTRAIN:   ({self._current_train_step}, {self._current_train_loss})' + \
            f'\n\t\t\t\tHOLDOUT: ({self._current_holdout_step}, {self._current_holdout_loss})')

        if any(tracker.has_data() for tracker in self._trackers.leaf_values()):
            tracker_str = "Trackers:"
            for tracker_name, tracker in self._trackers.leaf_items():
                if tracker.has_data():
                    #  the tracker has time series output (e.g. buffered returns), which we will average
                    ts_outputs = tracker.get_time_series().leaf_apply(lambda arr: np.asarray(arr)[None])  # T
                    writing_types = self._tracker_write_types[tracker_name]
                    for key, arr in ts_outputs.leaf_items():
                        if arr.size > 0:
                            tracker_str += f"\n------- {tracker_name} --------"
                            if 'mean' in writing_types:
                                tracker_str += f'\n{key + "_mean"}:  {arr.mean()}'
                            if 'max' in writing_types:
                                tracker_str += f'\n{key + "_max"}:  {arr.max()}'
                            if 'min' in writing_types:
                                tracker_str += f'\n{key + "_min"}:  {arr.min()}'
                            if 'std' in writing_types:
                                tracker_str += f'\n{key + "_std"}:  {arr.std()}'
            logger.debug(tracker_str)

        logger.debug(timeit)
        timeit.reset()

    # RUNNING SCRIPTS #

    def run_preamble(self):
        super(GoalTrainer, self).run_preamble()

        # checking for data if we are training with no env
        if self._step_train_env_every_n_steps == 0 and all(
                len(self._datasets_train[i]) == 0 for i in range(len(self._datasets_train))):
            raise Exception("Dataset is empty but no data is going to be collected")

        # intialize data save directory.
        for i in range(len(self._datasets_train)):
            if self._save_data_train_every_n_steps[i] > 0:
                self._datasets_train[i].create_save_dir()

        for i in range(len(self._datasets_holdout)):
            if self._save_data_holdout_every_n_steps[i] > 0:
                self._datasets_holdout[i].create_save_dir()

    def eval_step(self, obs_train, goal_train, obs_holdout, goal_holdout):
        if is_next_cycle(self._current_step, self._step_train_env_every_n_steps):
            with timeit('step train env'):
                for i in range(self._step_train_env_n_per_step):
                    obs_train, goal_train, _ = self.env_step(self._model, self._env_train,
                                                             self._datasets_train,
                                                             obs_train, goal_train, self._env_train_memory,
                                                             self._policy, self._goal_policy,
                                                             reward=self._reward,
                                                             eval=True,
                                                             trackers=self._trackers < ["env_train"],
                                                             curr_step=self._current_env_train_step,
                                                             add_data_every_n=self._add_to_data_train_every_n_goals)
                    self._current_env_train_step += 1
                    self._current_env_train_ep += int(self._env_train_memory.is_empty())

        if is_next_cycle(self._current_step, self._step_holdout_env_every_n_steps):
            with timeit('step holdout env'):
                for i in range(self._step_holdout_env_n_per_step):
                    obs_holdout, goal_holdout, _ = self.env_step(self._model, self._env_holdout,
                                                                 self._datasets_holdout,
                                                                 obs_holdout, goal_holdout,
                                                                 self._env_holdout_memory,
                                                                 self._policy_holdout,
                                                                 self._goal_policy_holdout,
                                                                 reward=self._reward, eval=True,
                                                                 trackers=self._trackers < ["env_holdout"],
                                                                 curr_step=self._current_env_holdout_step,
                                                                 add_data_every_n=self._add_to_data_holdout_every_n_goals)
                    self._current_env_holdout_step += 1
                    self._current_env_holdout_ep += int(self._env_holdout_memory.is_empty())

        if is_next_cycle(self._current_step, self._rollout_train_env_every_n_steps):
            with timeit('rollout train env'):
                for i in range(self._rollout_train_env_n_per_step):
                    step_wrapper = AttrDict(step=self._current_env_train_step)
                    obs_holdout, goal_holdout, _ = self.env_rollout(self._eval_model, self._env_train,
                                                                    self._datasets_train,
                                                                    obs_train, goal_train,
                                                                    self._env_train_memory,
                                                                    self._policy, self._goal_policy,
                                                                    reward=self._reward,
                                                                    trackers=self._trackers < ["env_train"],
                                                                    curr_step_wrapper=step_wrapper,
                                                                    add_to_data_every_n=self._add_to_data_train_every_n_goals)
                    self._current_env_train_step = step_wrapper["step"]
                    self._current_env_train_ep += 1

                # also force a write step here
                self._tracker_write_step(self._trackers < ["env_train"], self._current_env_train_step, force=True)

        if is_next_cycle(self._current_step, self._rollout_holdout_env_every_n_steps):
            with timeit('rollout holdout env'):
                for i in range(self._rollout_holdout_env_n_per_step):
                    step_wrapper = AttrDict(step=self._current_env_holdout_step)
                    obs_holdout, goal_holdout, _ = self.env_rollout(self._eval_model, self._env_holdout,
                                                                    self._datasets_holdout,
                                                                    obs_holdout, goal_holdout,
                                                                    self._env_holdout_memory,
                                                                    self._policy_holdout,
                                                                    self._goal_policy_holdout,
                                                                    reward=self._reward,
                                                                    trackers=self._trackers < [
                                                                        "env_holdout"],
                                                                    curr_step_wrapper=step_wrapper,
                                                                    add_to_data_every_n=self._add_to_data_holdout_every_n_goals)
                    self._current_env_holdout_step = step_wrapper["step"]
                    self._current_env_holdout_ep += 1

                # also force a write step here
                self._tracker_write_step(self._trackers < ["env_holdout"], self._current_env_holdout_step, force=True)

        return obs_train, goal_train, obs_holdout, goal_holdout

    def run(self, separate_eval=False):
        """
        This is the main loop:
            - gather data
            - train the model
            - save the model
            - log progress
        """
        self.run_preamble()

        # pre-training actions
        self._model.pretrain(datasets_holdout=self._datasets_holdout)

        obs_train = AttrDict()
        goal_train = AttrDict()
        obs_holdout = AttrDict()
        goal_holdout = AttrDict()

        # loop
        while self._current_step < self._max_steps:
            # NOTE: always have some form of timing so that you can find bugs
            with timeit('total_loop'):
                """ Model / Training actions """
                # UPDATES
                for i in range(len(self._datasets_train)):
                    if is_next_cycle(self._current_step, self._train_every_n_steps[i]) and len(
                            self._datasets_train[i]) > 0:
                        if self._current_step >= self._block_train_on_first_n_steps:
                            with timeit('train'):
                                self._train_step(self._model, i)
                                self._current_train_step[i] += 1

                # holdout actions
                for i in range(len(self._datasets_holdout)):
                    if is_next_cycle(self._current_step, self._holdout_every_n_steps[i]) and len(
                            self._datasets_holdout[i]) > 0:
                        with timeit('holdout'):
                            self._holdout_step(self._model, i)
                            self._current_holdout_step[i] += 1

                # update step (after taking the train step)
                self._current_step += 1

                """ Eval, saving, and logging actions """
                # environment rollouts
                if not separate_eval and self._current_step >= self._block_env_on_first_n_steps:
                    obs_train, goal_train, obs_holdout, goal_holdout = self.eval_step(obs_train, goal_train,
                                                                                      obs_holdout, goal_holdout)

                # SAVE MODEL
                if is_next_cycle(self._current_step, self._save_every_n_steps):
                    with timeit('save'):
                        do_best = False
                        # compare if tracker name is valid and new episodes have been rolled out.
                        if self._track_best_name in self._trackers.leaf_keys() and \
                                self._current_env_train_ep > self._last_best_env_train_ep:
                            tracker = self._trackers[self._track_best_name]
                            # if there's data, save if this is the best one yet.
                            if tracker.has_data():
                                curr_tracked_val = self._track_best_reduce_fn(
                                    np.asarray(tracker.get_time_series()[self._track_best_key]))
                                do_best = curr_tracked_val is not None and curr_tracked_val > self._last_best_tracked_val

                        if do_best:
                            self._last_best_tracked_val = curr_tracked_val
                            self._last_best_env_train_ep = self._current_env_train_ep
                            logger.info(
                                f"Saving best model: tracker = {self._track_best_name}, {self._track_best_key} = {curr_tracked_val}")

                            # also writing this only on saves.
                            if self._summary_writer is not None:
                                self._summary_writer.add_scalar(f"{self._track_best_name}/{self._track_best_key}_BEST",
                                                                self._last_best_tracked_val, self._current_step)

                        self._save(chkpt=is_next_cycle(self._current_step, self._save_checkpoint_every_n_steps),
                                   best=do_best)

                # SAVE DATA
                for i in range(len(self._datasets_train)):
                    if is_next_cycle(self._current_step, self._save_data_train_every_n_steps[i]):
                        with timeit('save_data_train'):
                            self._datasets_train[i].save()

                for i in range(len(self._datasets_holdout)):
                    if is_next_cycle(self._current_step, self._save_data_holdout_every_n_steps[i]):
                        with timeit('save_data_holdout'):
                            self._datasets_holdout[i].save()

            # log (outside of timeit)
            if is_next_cycle(self._current_step, self._log_every_n_steps):
                self._log()
