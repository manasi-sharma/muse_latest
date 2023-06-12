"""
This is taken from wilds repo
Supports group-wise and element-wise metrics over batches
"""
import numpy as np
import torch

from attrdict import AttrDict as d
from muse.utils.general_utils import is_array
from muse.utils.torch_utils import numel


class Metric:
    """
    Parent class for metrics.
    """

    def __init__(self, name):
        # metric name
        self._name = name

    def _compute(self, inputs: d, outputs: d, model_outputs: d):
        """
        Helper function for computing the metric.
        Subclasses should implement this.
        Args:
            - inputs (AttrDict of Tensor): Inputs
            - outputs (AttrDict of Tensor): True targets
            - model_outputs (AttrDict of Tensor): Predicted targets
        Output:
            - metric (0-dim tensor): metric
        """
        return NotImplementedError

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (0-dim tensor): Worst-case metric
        """
        raise NotImplementedError

    @property
    def name(self):
        """
        Metric name.
        Used to name the key in the results dictionaries returned by the metric.
        """
        return self._name

    @property
    def agg_metric_field(self):
        """
        The name of the key in the results dictionary returned by Metric.compute().
        This should correspond to the aggregate metric computed on all of y_pred and y_true,
        in contrast to a group-wise evaluation.
        """
        return f'{self.name}_all'

    def group_metric_field(self, group_idx):
        """
        The name of the keys corresponding to individual group evaluations
        in the results dictionary returned by Metric.compute_group_wise().
        """
        return f'{self.name}_group:{group_idx}'

    @property
    def worst_group_metric_field(self):
        """
        The name of the keys corresponding to the worst-group metric
        in the results dictionary returned by Metric.compute_group_wise().
        """
        return f'{self.name}_wg'

    def group_count_field(self, group_idx):
        """
        The name of the keys corresponding to each group's count
        in the results dictionary returned by Metric.compute_group_wise().
        """
        return f'count_group:{group_idx}'

    def compute(self, inputs: d, outputs: d, model_outputs: d, return_dict=True):
        """
        Computes metric. This is a wrapper around _compute.
        Args:
            - inputs (AttrDict of Tensor): Inputs
            - outputs (AttrDict of Tensor): True targets
            - model_outputs (AttrDict of Tensor): Predicted targets
            - return_dict (bool): Whether to return the output as a dictionary or a tensor
        Output (return_dict=False):
            - metric (0-dim tensor): metric. If the inputs are empty, returns tensor(0.)
        Output (return_dict=True):
            - results (dict): Dictionary of results, mapping metric.agg_metric_field to avg_metric
        """
        if outputs.is_empty():
            agg_metric = torch.tensor(0.)
        elif outputs.leaf_reduce(lambda red, arr: red and numel(arr) == 0, seed=True):
            agg_metric = torch.tensor(0., device=outputs.get_one().device)
        else:
            agg_metric = self._compute(inputs, outputs, model_outputs)
        if return_dict:
            results = d.from_dict({
                self.agg_metric_field: agg_metric
            })
            return results
        else:
            return agg_metric

    def compute_group_wise(self, inputs: d, outputs: d, model_outputs: d,  g, n_groups, return_dict=True):
        """
        Computes metrics for each group. This is a wrapper around _compute.
        Args:
            - inputs (AttrDict of Tensor): Inputs
            - outputs (AttrDict of Tensor): True targets
            - model_outputs (AttrDict of Tensor): Predicted targets
            - g (Tensor): groups
            - n_groups (int): number of groups
            - return_dict (bool): Whether to return the output as a dictionary or a tensor
        Output (return_dict=False):
            - group_metrics (Tensor): tensor of size (n_groups, ) including the average metric for each group
            - group_counts (Tensor): tensor of size (n_groups, ) including the group count
            - worst_group_metric (0-dim tensor): worst-group metric
            - For empty inputs/groups, corresponding metrics are tensor(0.)
        Output (return_dict=True):
            - results (dict): Dictionary of results
        """
        group_metrics, group_counts, worst_group_metric = \
            self._compute_group_wise(inputs, outputs, model_outputs, g, n_groups)
        if return_dict:
            results = d()
            for group_idx in range(n_groups):
                results[self.group_metric_field(group_idx)] = group_metrics[group_idx]
                results[self.group_count_field(group_idx)] = group_counts[group_idx]
            results[self.worst_group_metric_field] = worst_group_metric
            return results
        else:
            return group_metrics, group_counts, worst_group_metric

    def _compute_group_wise(self, inputs: d, outputs: d, model_outputs: d, g, n_groups):
        group_metrics = []
        group_counts = get_counts(g, n_groups)
        for group_idx in range(n_groups):
            if group_counts[group_idx] == 0:
                group_metrics.append(torch.tensor(0., device=g.device))
            else:
                group_metrics.append(
                    self._compute(
                        inputs.leaf_apply(lambda arr: arr[g == group_idx] if is_array(arr) else arr),
                        outputs.leaf_apply(lambda arr: arr[g == group_idx] if is_array(arr) else arr),
                        model_outputs.leaf_apply(lambda arr: arr[g == group_idx] if is_array(arr) else arr))
                )
        group_metrics = torch.stack(group_metrics)
        worst_group_metric = self.worst(group_metrics[group_counts > 0])

        return group_metrics, group_counts, worst_group_metric

    def __str__(self):
        return f"{type(self).__name__}(name={self._name})"


class ElementwiseMetric(Metric):
    """
    Averages.
    """

    def _compute_element_wise(self, inputs: d, outputs: d, model_outputs: d):
        """
        Helper for computing element-wise metric, implemented for each metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        """
        raise NotImplementedError

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (0-dim tensor): Worst-case metric
        """
        raise NotImplementedError

    def _compute(self, inputs: d, outputs: d, model_outputs: d):
        """
        Helper function for computing the metric.
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - avg_metric (0-dim tensor): average of element-wise metrics
        """
        element_wise_metrics = self._compute_element_wise(inputs, outputs, model_outputs)
        avg_metric = element_wise_metrics.mean()
        return avg_metric

    def _compute_group_wise(self, inputs: d, outputs: d, model_outputs: d, g, n_groups):
        element_wise_metrics = self._compute_element_wise(inputs, outputs, model_outputs)
        group_metrics, group_counts = avg_over_groups(element_wise_metrics, g, n_groups)
        worst_group_metric = self.worst(group_metrics[group_counts > 0])
        return group_metrics, group_counts, worst_group_metric

    @property
    def agg_metric_field(self):
        """
        The name of the key in the results dictionary returned by Metric.compute().
        """
        return f'{self.name}_avg'

    def compute_element_wise(self, inputs: d, outputs: d, model_outputs: d, return_dict=True):
        """
        Computes element-wise metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
            - return_dict (bool): Whether to return the output as a dictionary or a tensor
        Output (return_dict=False):
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        Output (return_dict=True):
            - results (dict): Dictionary of results, mapping metric.name to element_wise_metrics
        """
        element_wise_metrics = self._compute_element_wise(inputs, outputs, model_outputs)
        batch_size = outputs.get_one().size()[0]
        assert element_wise_metrics.dim() == 1 and element_wise_metrics.numel() == batch_size

        if return_dict:
            return d.from_dict({self.name: element_wise_metrics})
        else:
            return element_wise_metrics

    def compute_flattened(self, inputs: d, outputs: d, model_outputs: d, return_dict=True):
        flattened_metrics = self.compute_element_wise(inputs, outputs, model_outputs, return_dict=False)
        index = torch.arange(outputs.get_one().numel())
        if return_dict:
            return d.from_dict({self.name: flattened_metrics, 'index': index})
        else:
            return flattened_metrics, index


class MultiTaskMetric(Metric):
    def _compute_flattened(self, flattened_y_pred, flattened_y_true):
        raise NotImplementedError

    def _compute(self, inputs: d, outputs: d, model_outputs: d):
        flattened_metrics, _ = self.compute_flattened(inputs, outputs, model_outputs, return_dict=False)
        if flattened_metrics.numel() == 0:
            return torch.tensor(0., device=flattened_metrics.device)
        else:
            return flattened_metrics.mean()

    def _compute_group_wise(self, inputs: d, outputs: d, model_outputs: d, g, n_groups):
        flattened_metrics, indices = self.compute_flattened(y_pred, y_true, return_dict=False)
        flattened_g = g[indices]
        group_metrics, group_counts = avg_over_groups(flattened_metrics, flattened_g, n_groups)
        worst_group_metric = self.worst(group_metrics[group_counts > 0])
        return group_metrics, group_counts, worst_group_metric

    def compute_flattened(self, inputs: d, outputs: d, model_outputs: d, return_dict=True):
        is_labeled = ~torch.isnan(y_true)
        batch_idx = torch.where(is_labeled)[0]
        flattened_y_pred = y_pred[is_labeled]
        flattened_y_true = y_true[is_labeled]
        flattened_metrics = self._compute_flattened(flattened_y_pred, flattened_y_true)
        if return_dict:
            return {self.name: flattened_metrics, 'index': batch_idx}
        else:
            return flattened_metrics, batch_idx


class ExtractMetric(ElementwiseMetric):
    @property
    def agg_metric_field(self):
        """
        The name of the key in the results dictionary returned by Metric.compute().
        """
        return self._agg_metric_field

    def __init__(self, name, key=None, source=0, reduce_fn='none'):
        super(ExtractMetric, self).__init__(name)
        if key is None:
            key = name
        self._key = key
        self._source = source
        self._reduce_fn = reduce_fn.lower()
        assert self._reduce_fn in ['mean', 'max', 'min', 'none']
        self._agg_metric_field = f'{self.name}_{self._reduce_fn}' if self._reduce_fn != 'none' else str(self.name)
        assert 0 <= source < 3

    def _compute_element_wise(self, inputs: d, outputs: d, model_outputs: d):
        """
        Helper for computing element-wise metric, implemented for each metric
        Output:
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        """
        src = inputs if self._source == 0 else outputs
        src = model_outputs if self._source == 2 else src
        val = src[self._key]

        axes = list(range(1, len(val.shape)))
        if self._reduce_fn == 'mean':
            val = val.mean(axes)
        elif self._reduce_fn == 'max':
            val = val.max(axes)
        elif self._reduce_fn == 'min':
            val = val.min(axes)
        elif self._reduce_fn == 'none':
            pass
        else:
            raise NotImplementedError

        reshaped = val.reshape(-1)
        assert val.shape[0] == reshaped.shape[0], "Extracted metric"
        return reshaped

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (0-dim tensor): Worst-case metric
        """
        return minimum(metrics)
    
    def __str__(self):
        return f"{type(self).__name__}(name='{self._name}', key='{self._key}', source={self._source}, reduce_fn='{self._reduce_fn}')"


def avg_over_groups(v, g, n_groups):
    """
    Args:
        v (Tensor): Vector containing the quantity to average over.
        g (Tensor): Vector of the same length as v, containing group information.
    Returns:
        group_avgs (Tensor): Vector of length num_groups
        group_counts (Tensor)
    """
    import torch_scatter
    assert v.device == g.device, [v.device, g.device]
    device = v.device
    assert v.numel() == g.numel()
    group_count = get_counts(g, n_groups)
    group_avgs = torch_scatter.scatter(src=v, index=g, dim_size=n_groups, reduce='mean')
    return group_avgs, group_count


def get_counts(g, n_groups):
    """
    This differs from split_into_groups in how it handles missing groups.
    get_counts always returns a count Tensor of length n_groups,
    whereas split_into_groups returns a unique_counts Tensor
    whose length is the number of unique groups present in g.
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - counts (Tensor): A list of length n_groups, denoting the count of each group.
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    counts = torch.zeros(n_groups, device=g.device)
    counts[unique_groups] = unique_counts.float()
    return counts


def minimum(numbers, empty_val=0.):
    if isinstance(numbers, torch.Tensor):
        if numbers.numel() == 0:
            return torch.tensor(empty_val, device=numbers.device)
        else:
            return numbers[~torch.isnan(numbers)].min()
    elif isinstance(numbers, np.ndarray):
        if numbers.size == 0:
            return np.array(empty_val)
        else:
            return np.nanmin(numbers)
    else:
        if len(numbers) == 0:
            return empty_val
        else:
            return min(numbers)


def maximum(numbers, empty_val=0.):
    if isinstance(numbers, torch.Tensor):
        if numbers.numel() == 0:
            return torch.tensor(empty_val, device=numbers.device)
        else:
            return numbers[~torch.isnan(numbers)].max()
    elif isinstance(numbers, np.ndarray):
        if numbers.size == 0:
            return np.array(empty_val)
        else:
            return np.nanmax(numbers)
    else:
        if len(numbers) == 0:
            return empty_val
        else:
            return max(numbers)
