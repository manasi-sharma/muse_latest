"""
This is a file for loss and related functions that are shared between configs
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as D, nn as nn
from torch.nn import functional as F

from muse.utils.general_utils import timeit
from attrdict import AttrDict as d
from muse.utils.torch_utils import split_dim, to_torch, combine_after_last_dim, concatenate, combine_after_dim, \
    unsqueeze_then_gather, log_sum_exp


# -------------- Standard Element-wise Loss Definitions --------------

def mae_err_fn(pred, true, next=False, mask=None):
    H = pred.shape[-2]

    if mask is not None:
        assert mask.shape[-1] == true.shape[-2], [mask.shape, true.shape]

    m = (mask[..., -H:, None] if next else mask[..., :H, None]) if mask is not None else 1
    # factor normalizes the weights to ignore all zero elements (calling mean returns true mean.)
    factor = mask.numel() / mask.count_nonzero() if isinstance(mask, torch.Tensor) else 1.

    if next:
        return factor * m * torch.absolute(true[..., -H:, :] - pred)
    else:
        return factor * m * torch.absolute(true[..., :H, :] - pred)


def mse_err_fn(pred, true, next=False, mask=None):
    H = pred.shape[-2]

    if mask is not None:
        assert mask.shape[-1] == true.shape[-2], [mask.shape, true.shape]

    m = (mask[..., -H:, None] if next else mask[..., :H, None]) if mask is not None else 1
    # factor normalizes the weights to ignore all zero elements (calling mean returns true mean.)
    factor = mask.numel() / mask.count_nonzero() if isinstance(mask, torch.Tensor) else 1.

    if next:
        return factor * m * torch.square(true[..., -H:, :] - pred)
    else:
        return factor * m * torch.square(true[..., :H, :] - pred)


def sigmoid_mae_err_fn(pred, true, next=False):
    # assert pred.shape[-1] in [3, 2 + NUM_BLOCKS], "Only supports these action spaces (single or multiple grab)"
    H = pred.shape[-2]
    pred_grab = pred[..., 2:]

    # shifting
    if next:
        true = true[..., -H:]
    else:
        true = true[..., :H]

    true_grab = true[..., 2:]

    assert torch.logical_or(torch.isclose(true_grab, torch.zeros_like(true_grab)),
                            torch.isclose(true_grab, torch.ones_like(true_grab))).all(), "binary options"
    ae = torch.absolute(true - pred)
    grab_n1_or_p1 = 2 * true_grab - 1  # -1 or 1, binary
    # if grab == 1, its a flipped sigmoid (loss = 1 when grab < 0)
    ae[..., 2:] = torch.sigmoid(-grab_n1_or_p1 * pred_grab)
    return ae


def get_scaled_mae_err_fn(scale, device):
    # scales the last dim
    scale = to_torch(scale, device=device).to(dtype=torch.float32)

    def scaled_mae_err_fn(pred, true, next=False, mask=None):
        new_shape = np.array(list(pred.shape))
        new_shape[:-1] = 1
        multiplier = torch.broadcast_to(scale, new_shape.tolist())
        return multiplier * mae_err_fn(pred, true, next=False, mask=None)

    return scaled_mae_err_fn


def final_n_mae_err_fn(pred, true, final_n=1, next=False, adjust_scale=True):
    H = pred.shape[-2]
    if next:
        diff = torch.absolute(true[..., -H:, :] - pred)
    else:
        diff = torch.absolute(true[..., :H, :] - pred)

    # no errors on the last N dimensions
    assert diff.shape[-2] > final_n > 0, [diff.shape, final_n]
    diff[..., :-final_n, :] = 0
    if adjust_scale:
        # this is if we might take the mean over the horizon axis, for example. it upweights nonzero elements.
        diff = diff * diff.shape[-2] / final_n
    return diff


elementwise_loss_map = {
    'l1': mae_err_fn,
    'l2': mse_err_fn,
    'final_l1': final_n_mae_err_fn,
}


# -------------- Non-Element-wise Loss Definitions --------------

def pose_mae_err_fn(pred, true, next=False, mask=None):
    # pose_mae_err_fn
    # Euler angles in pred & true are unnormalized
    # put these in the range of [-np.pi, np.pi], which is what pred is.
    true[..., 3:6] = (true[..., 3:6] + np.pi) % (2 * np.pi) - np.pi
    err = mae_err_fn(pred, true, next=next, mask=mask)
    assert err.shape[-1] in [6, 7], err.shape  # grab action

    # this will be unique bc of how these angles are generated.
    orn_slice = err[..., 3:6]
    orn_slice = torch.relu(torch.minimum(orn_slice, 2 * np.pi - orn_slice))
    # forcing the error to be normalized

    err = torch.cat([
        err[..., :3],
        orn_slice[..., 0:1] / np.pi,  # 0 -> 1
        orn_slice[..., 1:2] / np.pi / 2,  # 0 -> 1
        orn_slice[..., 2:3] / np.pi,  # 0 -> 1
        err[..., 6:],
    ], dim=-1)
    return err


# -------------- OTHER --------------

# project tensor a (..., dim) onto b (..., dim)
def residual(a, b):
    coef = (a * b).sum(-1) / (torch.norm(b, dim=-1) ** 2 + 1e-14)
    return a - coef.unsqueeze(-1) * b


def dist_fro(inp, start=2, sqrt=False):
    d = (inp ** 2).sum(dim=tuple(range(start, len(inp.shape))))
    if sqrt:
        return d.sqrt()
    else:
        return d


# reconstruction L2 loss
def reconstruction_L2(x, xmean, reduce_range=(1, 4)):
    return - ((x - xmean) ** 2).sum(dim=tuple(range(reduce_range[0], reduce_range[1])))


# KL divergence btwn a given distribution and Normal(0,1)
def kl_divergence_with_normal(mean, cov):
    v1 = torch.diagonal(cov, dim1=-2, dim2=-1)  # (N,zd)
    return 0.5 * (-v1.log().sum(-1) + v1.sum(-1) + mean.norm(dim=-1) ** 2 - v1.shape[-1])

def kl_divergence_with_normal_dist(dist):
    mean = dist.mean
    v1 = dist.noise_std
    return 0.5 * (-v1.log().sum(-1) + v1.sum(-1) + mean.norm(dim=-1) ** 2 - v1.shape[-1])

# linefit loss
def linefit_loss(z, H):  # z must be (N*(H+1), zd1...)
    N = z.shape[0] // (H + 1)
    z = split_dim(z, 0, [N, H + 1])  # (N, H+1, z1..)
    z1 = z[:, :1]
    z2 = z[:, -1:]
    res = residual(z - z1, z2 - z1)

    return dist_fro(res)

def pointfit_loss(z, H):
    N = z.shape[0] // (H + 1)
    z = z.view([N, H+1] + list(z.shape[1:]))  # (N, H+1, z1..)
    
    z1 = z[:, :1]
    z2 = z[:, -1:] 
    z_shape = z.shape
    z_all = z1
    for i in range(1, z_shape[1]):
          z_i = i/(z_shape[1]-1)*z2 + (1-i/(z_shape[1]-1))*z1
          z_all = torch.cat([z_all, z_i], dim=1)
    res = (z_all - z)

    pointfit = (res.norm(dim=-1) ** 2).sum(-1)
    return pointfit


def interpolated_recon_loss(z, H, x, model):
    # x should not be preprocessed
    N = z.shape[0] // (H + 1)
    z = split_dim(z, 0, [N, H + 1])
    z1 = z[:, :1]
    z2 = z[:, -1:] 
    z_shape = z.shape
    z_all = z1
    for i in range(1, H + 1):
        z_i = i/H * z2 + (1-i/H) * z1
        z_all = torch.cat([z_all, z_i], dim=1)
    z_all = z_all.view([N * (H+1)] + list(z.shape[2:]))

    interpolated_x_mean = model.decode_forward(z_all, postproc=True)
    interpolated_recon = -((interpolated_x_mean - x)**2).sum([-1, -2, -3])
    return interpolated_recon.view(N, H+1)


# triangle loss
def triangle_loss(z, H):
    N = z.shape[0] // (H + 1)
    z = split_dim(z, 0, [N, H + 1])  # (N, H+1, z1..)
    z1 = z[:, :1]
    z2 = z[:, -1:]

    # (N, H)
    d = dist_fro(z[:, 1:] - z[:, :-1], start=2, sqrt=True) - dist_fro(z1 - z2, start=2, sqrt=True).detach() / H
    return d


def binary_logits_to_score(logits):
    assert logits.dim() in (1, 2)
    if logits.dim() == 2: #multi-class logits
        assert logits.size(1) == 2, "Only binary classification"
        score = F.softmax(logits, dim=1)[:, 1]
    else:
        score = logits
    return score


def multiclass_logits_to_pred(logits):
    """
    Takes multi-class logits of size (batch_size, ..., n_classes) and returns predictions
    by taking an argmax at the last dimension
    """
    assert logits.dim() > 1
    return logits.argmax(-1)


def binary_logits_to_pred(logits):
    return (logits>0).long()


def write_avg_per_last_dim(ae, i, writer, writer_prefix):
    dms = len(ae.shape)
    ae_mean_per_dim = ae.mean(dim=tuple(range(dms - 1)))
    for dim in range(len(ae_mean_per_dim)):
        writer.add_scalar(writer_prefix + "%d" % dim, ae_mean_per_dim[dim].item(), i)


# --------------- other, train compatible, loss functions --------------------


def get_default_mae_object_loss_fn(policy_out_names, err_fn=mae_err_fn, use_outs=False, mask_name=None):
    to_normalize = list(policy_out_names)

    def mae_object_loss_fn(model, policy_out: d, ins: d, outs: d, i=0, writer=None, writer_prefix="", normalize=True,
                           **kwargs):
        # H-1
        relevant = policy_out.node_leaf_filter_keys_required(policy_out_names)
        normalized_pred = concatenate(combine_after_last_dim(relevant), policy_out_names, dim=-1)

        source = (outs if use_outs else ins).leaf_copy()
        # B x H-1, padding mask
        loss_mask = source << mask_name if mask_name is not None else None
        loss_mask = ~loss_mask if loss_mask is not None else None

        # true
        if normalize:
            source = model.normalize_by_statistics(source, to_normalize)
        relevant_true = source.node_leaf_filter_keys_required(policy_out_names)
        # H
        normalized_true = concatenate(combine_after_last_dim(relevant_true), policy_out_names,
                                      dim=-1)  # concatenate losses to compute over all dims
        err = err_fn(normalized_pred, normalized_true, mask=loss_mask, next=True)

        if writer is not None:
            write_avg_per_last_dim(err, i=i, writer=writer, writer_prefix=writer_prefix + "object_loss/mae_dim_")

            # log the avg delta of the true
            traj_delta = (normalized_true[:, 1:] - normalized_true[:, :-1]).abs().sum(1) / normalized_true.shape[1]
            write_avg_per_last_dim(traj_delta, i=i, writer=writer,
                                   writer_prefix=writer_prefix + "object_loss/true_l1_dist_dim_")

            # log the delta of the pred
            traj_delta = (normalized_pred[:, 1:] - normalized_pred[:, :-1]).abs().sum(1) / normalized_pred.shape[1]
            write_avg_per_last_dim(traj_delta, i=i, writer=writer,
                                   writer_prefix=writer_prefix + "object_loss/pred_l1_dist_dim_")

        return err.mean(dim=-1)

    return mae_object_loss_fn


def get_default_nll_loss_fn(policy_out_names, policy_dist_name="policy_raw", use_outs=False, relative=False,
                            policy_out_norm_names=None, vel_act=False,
                            preproc_true_fn=None):
    if policy_out_norm_names is None:
        policy_out_norm_names = list(policy_out_names)
    to_normalize = list(policy_out_norm_names)

    if relative:
        raise NotImplementedError

    def loss_fn(model, policy_out: d, ins: d, outs: d, i=0, writer=None, writer_prefix="", normalize=True, **kwargs):
        action_dist = policy_out[policy_dist_name]
        # H
        source = outs if use_outs else ins
        if normalize:
            source = model.normalize_by_statistics(source, to_normalize)
        # first H-1 from H, these are the actions we care about
        relevant_true = (source > policy_out_names)
        # relevant_true.leaf_modify(lambda arr: arr[:, :-1])  # truncate manually.

        ac_mean_horizon = action_dist.mean.shape[1]
        # logger.debug(f"ac mean shape = {action_dist.mean.shape}")
        # relevant_true.leaf_shapes().pprint()

        relevant_true.leaf_assert(lambda arr: arr.shape[1] == ac_mean_horizon)  # make sure horizons match
        relevant_true.leaf_modify(lambda arr: combine_after_dim(arr, 2))
        in_arr = concatenate(relevant_true, policy_out_names, dim=2)  # B x H x sumD

        if preproc_true_fn is not None:
            in_arr = preproc_true_fn(in_arr)

        # event dim =0, so shape is (B x H x sumD)
        nll = -action_dist.log_prob(in_arr)

        # only write if there is a non-horizon last dimension here.
        if writer is not None and len(nll.shape) >= 3:
            write_avg_per_last_dim(nll, i=i, writer=writer, writer_prefix=writer_prefix + "policy_loss/nll_dim_")

        if len(nll.shape) > 2:
            nll = nll.mean(dim=-1)
        return nll

    return loss_fn


def get_default_mae_action_loss_fn(policy_out_names, max_grab=None, err_fn=mae_err_fn, use_outs=False, relative=False,
                                   mask_name=None, vel_act=False, policy_out_norm_names=None, closest_block_pos=None):
    to_normalize = list(policy_out_names) if policy_out_norm_names is None else list(policy_out_norm_names)
    if vel_act:
        assert max_grab is None
        assert not relative, "not implemented yet"

    if max_grab is not None:
        assert "target/grab_binary" in policy_out_names
        to_normalize.remove("target/grab_binary")

    if relative:
        assert "target/position" in policy_out_names

    def mae_action_loss_fn(model, policy_out: d, ins: d, outs: d, i=0, writer=None, writer_prefix="", normalize=True,
                           **kwargs):
        # H-1
        normalized_pred = concatenate(policy_out, policy_out_names, dim=-1)

        source = (outs if use_outs else ins).leaf_copy()
        # B x H-1, padding mask
        loss_mask = source << mask_name if mask_name is not None else None
        loss_mask = ~loss_mask if loss_mask is not None else None

        if relative:
            if policy_out.has_node_leaf_key("sampler"):  # sampler was used to get actions
                # sampler is normalized?
                sampled_ins_unnorm = model.normalize_by_statistics(policy_out["sampler"],
                                                                   ['block_positions', 'position'], inverse=True)
                close_bpos = closest_block_pos(sampled_ins_unnorm)
            else:
                close_bpos = closest_block_pos(ins)
            # truncate to input shape before subtraction
            close_bpos = close_bpos[:, :(source["target/position"]).shape[1]]
            # print(close_bpos.shape, source['target/position'].shape)
            # relatget_default_lmp_names_and_sizesive to nearest block, predict the delta, might make normalization slightly off...
            source['target/position'] = (source['target/position']) - close_bpos

        if normalize:
            source = model.normalize_by_statistics(source, to_normalize)
        if max_grab is not None:
            source['target/grab_binary'] = (source["target/grab_binary"]) / max_grab
        # H
        normalized_true = concatenate(source, policy_out_names, dim=-1)  # concatenate losses to compute over all dims
        if use_outs:
            assert normalized_true.shape[1] == normalized_pred.shape[
                1], f"horizons don't match, true shape: {normalized_true.shape}, pred: {normalized_pred.shape}"

        # compute the error
        # print(loss_mask.shape if loss_mask is not None else None, normalized_true.shape)
        err = err_fn(normalized_pred, normalized_true, mask=loss_mask)

        if writer is not None:
            write_avg_per_last_dim(err, i=i, writer=writer, writer_prefix=writer_prefix + "policy_loss/mae_dim_")
        return err.mean(dim=-1)

    return mae_action_loss_fn


def get_info_nce_loss_fn(score_name, negative_score_name, autoregressive):
    def mae_nce_loss_fn(model, energy_model_out: d, ins: d, outs: d, i=0, writer=None, writer_prefix="", normalize=True,
                        ret_dict=False, **kwargs):
        # positive: (..., 1), or last_dim=|A| if autoregressive
        energy = energy_model_out[score_name]
        # negative: (..., N, 1), or last_dim=|A| if autoregressive
        negative_score = energy_model_out[negative_score_name]

        # make pos same shape as neg
        # exp_s = torch.exp(-score)
        # exp_ns = torch.exp(-negative_score)

        # (..., 1), or last_dim=|A| if autoregressive
        stacked_energy = torch.cat([energy[..., None, :], negative_score], dim=-2)
        # info nce: -log(exp(-pos) / ( exp(-pos) + SUM < exp(-neg) > ) = - (-pos - log( exp(-pos) + SUM < exp(-neg) > ))
        loss = energy + torch.logsumexp(-stacked_energy, dim=-2)

        if writer is not None:
            if autoregressive:
                write_avg_per_last_dim(energy, i, writer, writer_prefix + "positives_energy/dim_")
                write_avg_per_last_dim(negative_score, i, writer, writer_prefix + "negatives_energy/dim_")
                write_avg_per_last_dim(loss, i, writer, writer_prefix + "loss/dim_")

            with timeit("writer"):
                writer.add_scalar(writer_prefix + "positives_energy", energy[..., -1].mean().item(), i)
                writer.add_scalar(writer_prefix + "negatives_energy", negative_score[..., -1].mean().item(), i)
                writer.add_scalar(writer_prefix + "min_energy", stacked_energy[..., -1].min().item(), i)
                writer.add_scalar(writer_prefix + "max_energy", stacked_energy[..., -1].max().item(), i)
                writer.add_scalar(writer_prefix + "loss", loss[..., -1].mean().item(), i)

        if ret_dict:
            return d(loss=loss)

        return loss

    return mae_nce_loss_fn


def get_kmd_loss_fn(policy_out_names, IDX_NAME, ALL_PROB_NAME, RES_NAME, ALL_RESIDUAL_NAME, policy_out_norm_names=None,
                    use_outs=False, focal=1., beta=1., ):
    loss_obj = FocalLoss(focal)
    res_err_fn = mse_err_fn

    def kmd_loss_fn(model, policy_out: d, ins: d, outs: d, i=0, writer=None, writer_prefix="", normalize=True,
                    **kwargs):

        source = outs if use_outs else ins
        disc_out = model.discretize.forward(source)
        true_idxs = disc_out[IDX_NAME]  # B x H
        pred_prob = policy_out[ALL_PROB_NAME]  # B x H x C

        B, H = true_idxs.shape[:2]

        true_residual: d = disc_out[RES_NAME]
        all_residuals: d = policy_out[ALL_RESIDUAL_NAME]  # B x H x C x |A|

        # focal loss over class dimension (2). Cross entropy expects prob = B x C and target = B
        focal_loss = loss_obj(pred_prob.view(B * H, -1), true_idxs.reshape(B * H, )).mean()

        # residual loss
        norm_pred_residual = all_residuals.leaf_apply(
            lambda arr: unsqueeze_then_gather(arr, true_idxs, len(true_idxs.shape)))
        norm_true_residual = model.normalize_by_statistics(true_residual, policy_out_norm_names)

        res_loss = []
        for key in policy_out_names:
            res_loss.append(combine_after_dim(res_err_fn(norm_pred_residual[key], norm_true_residual[key]), 2))

            if writer is not None:
                with timeit("writer"):
                    writer.add_scalar(writer_prefix + f"res_loss/{key}", res_loss[-1].mean(), i)

        res_loss = torch.cat(res_loss, dim=-1).mean()
        loss = focal_loss + beta * res_loss

        if writer is not None:
            with timeit("writer"):
                writer.add_scalar(writer_prefix + "beta", beta, i)
                writer.add_scalar(writer_prefix + "loss", loss.item(), i)
                writer.add_scalar(writer_prefix + "focal_loss", focal_loss.item(), i)
                writer.add_scalar(writer_prefix + "res_loss", res_loss.item(), i)

        return loss

    return kmd_loss_fn


def log_gaussian_prob(mu_obs, sigma_obs, targ_obs):
    assert mu_obs.shape == sigma_obs.shape == targ_obs.shape, "%s, %s, %s" % \
                                                              (mu_obs.shape, sigma_obs.shape, targ_obs.shape)
    # assume last dimension is N
    N = mu_obs.shape[-1]

    det = (sigma_obs ** 2).prod(dim=-1)  # should be (batch, num_models) or just (batch,)
    a = torch.log((2 * np.pi) ** N * det)
    b = ((targ_obs - mu_obs) ** 2 / sigma_obs ** 2).sum(-1)  # (batch, nm, 3) -> (batch, nm) or without num_models

    assert a.shape == b.shape

    # cov determinant term + scaled squared error
    return - 0.5 * (a + b).mean()  # mean over batch and num models


def kl_regularization(latent_mu, latent_log_sigma, mean_p=0., sigma_p=1.):
    var_q = (latent_log_sigma.exp()) ** 2
    var_p = sigma_p ** 2
    sigma_p = torch.tensor(sigma_p, device=latent_mu.device)
    kl = ((latent_mu - mean_p) ** 2 + var_q) / (2. * var_p) + (torch.log(sigma_p) - latent_log_sigma) - 0.5
    return kl.mean()


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def discretized_mix_logistic_loss(y_hat, y, num_classes=256,
                                  log_scale_min=-7.0, reduce=True):
    """Discretized mixture of logistic distributions loss

    Note that it is assumed that input is scaled to [-1, 1].

    Args:
        y_hat (Tensor): Predicted output (B x C x T)
        y (Tensor): Target (B x T x 1).
        num_classes (int): Number of classes
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each
          minibatch.

    Returns
        Tensor: loss
    """
    assert y_hat.dim() == 3
    assert y_hat.size(1) % 3 == 0
    nr_mix = y_hat.size(1) // 3

    # (B x T x C)
    y_hat = y_hat.transpose(1, 2)

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)

    # B x T x 1 -> B x T x num_mixtures
    y = y.expand_as(means)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(torch.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - torch.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = (cdf_delta > 1e-5).float()

    inner_inner_out = inner_inner_cond * \
                      torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
                      (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    inner_cond = (y > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (y < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        return -torch.sum(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def mix_gaussian_loss(y_hat, y, log_scale_min=-7.0, reduce=True):
    """Mixture of continuous gaussian distributions loss

    Note that it is assumed that input is scaled to [-1, 1].

    Args:
        y_hat (Tensor): Predicted output (B x C x T)
        y (Tensor): Target (B x T x 1).
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each
          minibatch.
    Returns
        Tensor: loss
    """
    assert y_hat.dim() == 3
    C = y_hat.size(1)
    if C == 2:
        nr_mix = 1
    else:
        assert y_hat.size(1) % 3 == 0
        nr_mix = y_hat.size(1) // 3

    # (B x T x C)
    y_hat = y_hat.transpose(1, 2)

    # unpack parameters.
    if C == 2:
        # special case for C == 2, just for compatibility
        logit_probs = None
        means = y_hat[:, :, 0:1]
        log_scales = torch.clamp(y_hat[:, :, 1:2], min=log_scale_min)
    else:
        #  (B, T, num_mixtures) x 3
        logit_probs = y_hat[:, :, :nr_mix]
        means = y_hat[:, :, nr_mix:2 * nr_mix]
        log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)

    # B x T x 1 -> B x T x num_mixtures
    y = y.expand_as(means)

    centered_y = y - means
    dist = D.Normal(loc=0., scale=torch.exp(log_scales))
    # do we need to add a trick to avoid log(0)?
    log_probs = dist.log_prob(centered_y)

    if nr_mix > 1:
        log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        if nr_mix == 1:
            return -torch.sum(log_probs)
        else:
            return -torch.sum(log_sum_exp(log_probs))
    else:
        if nr_mix == 1:
            return -log_probs
        else:
            return -log_sum_exp(log_probs).unsqueeze(-1)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets
    From: https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b
    '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' \
            else torch.sum(loss) if self.reduction == 'sum' \
            else loss
