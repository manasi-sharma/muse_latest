import numpy as np
import torch
from torch.distributions import kl_divergence

from muse.models.function_model import FunctionModel
from muse.models.layers.linear_vq_vae import vq_unsupervised_losses
from muse.models.dist.helpers import detach_normal, upper_kl_normal_softmaxmix_indnormal
from muse.utils.loss_utils import kl_divergence_with_normal_dist
from attrdict import AttrDict as d
from muse.utils.torch_utils import broadcast_dims, concatenate, combine_after_dim, to_torch


def concat_goal_select_fn(pl_inputs: d, input_names, goal_state_names, cat_dim=1, allow_missing=False,
                          prefix="goal_states"):
    # concats the state with the goal (only for goal_state_names)
    inps = pl_inputs > input_names  # (B x H x ...)
    H = inps.get_one().shape[1]
    goals = (pl_inputs > [f"{prefix}/{g}" for g in goal_state_names]).leaf_apply(lambda arr:
                                                                                    broadcast_dims(
                                                                                        arr,
                                                                                        [1], [H])
                                                                                    )  # (B x ...) -> (B x H x ...)
    if not goals.is_empty():
        goals = goals[prefix]

    if allow_missing:
        # allow_missing = True -> goal_states can be a superset of input_names.
        goals, extra_inps = goals.leaf_partition(lambda k, v: k in inps.list_leaf_keys())
        inps.combine(extra_inps)
    # state then goal, for only goal state names
    out = inps & d.leaf_combine_and_apply([goals, inps], lambda vs: torch.cat(vs[::-1], dim=cat_dim), match_keys=False)

    return out


def cut_then_concat_goal_select_fn(pl_inputs: d, input_names, goal_state_names, cut_last_h_steps=None, cat_dim=1,
                                   allow_missing=False, prefix="goal_states"):
    all_inps = pl_inputs.node_leaf_filter_keys_required(input_names + [f"{prefix}/" + g for g in goal_state_names])
    all_inps = all_inps & (pl_inputs < ["padding_mask"])  # add padding mask to inputs if present
    # cut from back
    if cut_last_h_steps is not None and cut_last_h_steps != 0:
        all_inps.leaf_assert(lambda arr: arr.shape[1] >= cut_last_h_steps)
        all_inps.leaf_modify(lambda arr:
                             arr[:, :-cut_last_h_steps] if cut_last_h_steps != 1 or arr.shape[1] > 1 else arr)
    return all_inps & concat_goal_select_fn(all_inps, input_names, goal_state_names, cat_dim=cat_dim,
                                            allow_missing=allow_missing, prefix=prefix)


def get_policy_type_switch(pl_inputs: d, type_key):
    ptype = combine_after_dim(pl_inputs[type_key], start_dim=2)
    assert ptype.shape[2] == 1
    true_type_diffs = (torch.diff(ptype[..., 0], dim=1).abs() > 1e-4).any(dim=1)
    return true_type_diffs


def get_gcbc_preproc_fn(no_goal, use_final_goal, device, POLICY_NAMES, POLICY_GOAL_STATE_NAMES):
    def policy_preproc_fn(inputs):
        # move inputs and potentially specified goals to torch to the right dtype.
        p_in = inputs.node_leaf_filter_keys_required(POLICY_NAMES) & (inputs < ["goal_states", "goal"])
        p_in = p_in.leaf_apply(
            lambda arr: to_torch(arr, device=device, check=True).to(dtype=torch.float32))
        H = (inputs[POLICY_NAMES[0]]).shape[1]

        if not no_goal:
            if use_final_goal:
                # use the last provided goal/state as the goal. uses {goal} as the prefix.
                assert "goal" in inputs.keys(), "GOAL KEYS must be present if using final goal."
                p_in = concat_goal_select_fn(p_in, POLICY_NAMES, POLICY_GOAL_STATE_NAMES, cat_dim=-1,
                                             allow_missing=False, prefix="goal")
            elif "goal_states" in inputs.keys():
                # looks for goal_states if present.
                p_in = concat_goal_select_fn(p_in, POLICY_NAMES, POLICY_GOAL_STATE_NAMES, cat_dim=-1,
                                             allow_missing=False)
            else:
                # if goal_states is not present, fill it in by selecting the last state in the window.
                assert H > 1, "must have at least two horizon elements if goal_states are not provided"
                goals = (p_in > POLICY_GOAL_STATE_NAMES).leaf_apply(
                    lambda arr: broadcast_dims(arr[:, -1:], [1], [H]))  # last state
                p_in.safe_combine(goals.leaf_key_change(lambda k, v: f'goal_states/{k}'))
                p_in = cut_then_concat_goal_select_fn(p_in, POLICY_NAMES, POLICY_GOAL_STATE_NAMES,
                                                      cut_last_h_steps=1,
                                                      cat_dim=-1)

        # flatten things before the model (shape will be B x H x Di for key i)
        p_in.leaf_modify(lambda arr: combine_after_dim(arr, 2))

        return p_in

    return policy_preproc_fn


def get_goal_preproc_fn(NAMES, GOAL_STATE_NAMES):
    """
    prepares goals to be concatenated to the input dimension, for keys that are shared (batches horizon)
    """
    def goal_preproc_fn(inputs):
        # B x H x D
        post_in = (inputs > NAMES).leaf_apply(lambda arr: combine_after_dim(arr, 2))
        seq_len = post_in.get_one().shape[1]
        # B x H x PRD
        goal = ((inputs["goal_states"]) > GOAL_STATE_NAMES).leaf_apply(
            lambda arr: broadcast_dims(combine_after_dim(arr, 2), [1], [seq_len]))
        return post_in & d(goal_states=goal)
    return goal_preproc_fn


def get_policy_preproc_fn(plan_name, policy_names, zero_out_policy_plan=False):
    """ POLICY preprocessing fn, which optionally zeroes the plan, but also flattens the inputs after horizon dim. """
    def policy_preproc_fn(inputs):
        p_in = inputs.leaf_filter_keys(policy_names)
        p_in.leaf_modify(lambda arr: combine_after_dim(arr, 2))
        if zero_out_policy_plan:
            p_in[plan_name] = 0. * p_in[plan_name]
        return inputs & p_in

    return policy_preproc_fn


# --------------------- PLAN distances --------------------- #


def get_plan_dist_fn(plan_name, use_gmm_prior=False):
    """ KL divergence between gaussians (maybe GMM's) """
    if use_gmm_prior:
        def plan_fn(prior, posterior, ins, outs, i=0, writer=None, writer_prefix="", **kwargs):
            base_kl = upper_kl_normal_softmaxmix_indnormal(
                posterior[plan_name + "_dist"],
                prior[plan_name + "_dist"], temp=np.inf)

            if writer is not None:
                # weights. these are purely for debugging purposes.
                writer.add_scalar(writer_prefix + f"plan_distance/base_kl", base_kl.mean().item(), i)
                if use_gmm_prior:
                    entropy = prior[plan_name + "_dist"].mixture_distribution.entropy()  # B x K
                    comp_entropy = prior[plan_name + "_dist"].component_distribution.entropy()  # B x K x D
                    writer.add_scalar(writer_prefix + f"plan_distance/prior_mix_entropy", entropy.mean().item(), i)
                    writer.add_scalar(writer_prefix + f"plan_distance/prior_comp_entropy", comp_entropy.mean().item(),
                                      i)

            return base_kl

        return plan_fn
        # return lambda prior, posterior, ins, outs, **kwargs: upper_kl_normal_softmaxmix_indnormal(
        #     posterior[plan_name + "_dist"],
        #     prior[plan_name + "_dist"], temp=np.inf)
    return lambda prior, posterior, ins, outs, **kwargs: kl_divergence(posterior[plan_name + "_dist"],
                                                                       prior[plan_name + "_dist"])


def get_fixed_posterior_plan_dist_fn(plan_name, fixed_beta=1e-3, use_gmm_prior=False, block_prior_steps=0,
                                     gmm_logprob=False):
    """ KL, but posterior doesn't get gradient updates. """
    def plan_fn(prior, posterior, ins, outs, i=0, writer=None, writer_prefix="", **kwargs):
        # detach posterior for the base_kl, as to not propagate gradients.
        if use_gmm_prior:
            if gmm_logprob:
                # NLL, not really a KL, but for consistency
                base_kl = -prior[plan_name + "_dist"].log_prob(posterior[plan_name + "_dist"].sample())
            else:
                base_kl = upper_kl_normal_softmaxmix_indnormal(
                    detach_normal(posterior[plan_name + "_dist"]),
                    prior[plan_name + "_dist"], temp=np.inf)
        else:
            base_kl = kl_divergence(detach_normal(posterior[plan_name + "_dist"]),
                                    prior[plan_name + "_dist"])

        fixed_kl = kl_divergence_with_normal_dist(posterior[plan_name + "_dist"])

        if writer is not None:
            # weights. these are purely for debugging purposes.
            if gmm_logprob:
                writer.add_scalar(writer_prefix + f"plan_distance/base_nll", base_kl.mean().item(), i)
            else:
                writer.add_scalar(writer_prefix + f"plan_distance/base_kl", base_kl.mean().item(), i)
            writer.add_scalar(writer_prefix + f"plan_distance/fixed_kl", fixed_kl.mean().item(), i)
            writer.add_scalar(writer_prefix + f"plan_distance/fixed_beta", fixed_beta, i)
            if use_gmm_prior:
                entropy = prior[plan_name + "_dist"].mixture_distribution.entropy()  # B x K
                comp_entropy = prior[plan_name + "_dist"].component_distribution.entropy()  # B x K x D
                writer.add_scalar(writer_prefix + f"plan_distance/prior_mix_entropy", entropy.mean().item(), i)
                writer.add_scalar(writer_prefix + f"plan_distance/prior_comp_entropy", comp_entropy.mean().item(), i)

        loss = fixed_beta * fixed_kl
        if i >= block_prior_steps:
            # combination of prior loss (detached from posterio) and weak posterior loss (normalized to N(0,1))
            return loss + base_kl
        else:
            # only the weak posterior loss
            return loss

    return plan_fn


def get_vq_plan_dist_fn(plan_name, vq_beta=0.25, beta_post=1., beta_pr=1.):
    """ VectorQuantize plan distances """
    # VQ beta is the standard VQ VAE beta. (embedding loss) + beta * (encoding loss)
    # Beta Posterior is the weighting of the VQ unsupervised objective ^^ to the prior VQ unsupervised objective.
    def vq_plan_fn(prior, posterior, ins, outs, i=0, writer=None, writer_prefix="", **kwargs):
        # prior is discrete.
        tup = posterior[plan_name + "_vq_tuple"]  # tuple
        posterior_dc = d(quantize=tup[0], input=tup[1], quantize_input_grad=tup[2], embed_idxs=tup[3], embed=tup[4])
        # run vq "forward" with the posterior selected code, but prior input
        prior_tup = prior[plan_name + "_vq_tuple"]  # in
        prior_quantize_input_grad = prior_tup[1] + (tup[0] - prior_tup[1]).detach()
        prior_dc = d(quantize=tup[0], input=prior_tup[1], quantize_input_grad=prior_quantize_input_grad,
                     embed_idxs=tup[3], embed=tup[4])
        # embedding dist + beta * encoder dist
        posterior_losses = vq_unsupervised_losses(posterior_dc, beta=vq_beta)
        prior_losses = vq_unsupervised_losses(prior_dc, beta=vq_beta)

        # this loss is not explicitly used.
        prior_post_distance = (prior_dc["input"] - posterior_dc["input"]).pow(2).mean(-1)
        if writer is not None:
            # weights. these are purely for debugging purposes.
            writer.add_scalar(writer_prefix + f"plan_distance/vq_beta", vq_beta, i)
            writer.add_scalar(writer_prefix + f"plan_distance/vq_beta_posterior", beta_post, i)

            # loss tensors.
            for key, item in posterior_losses.leaf_items():
                writer.add_scalar(writer_prefix + f"plan_distance/posterior/{key}", item.mean().item(), i)
            for key, item in prior_losses.leaf_items():
                writer.add_scalar(writer_prefix + f"plan_distance/prior/{key}", item.mean().item(), i)
            writer.add_scalar(writer_prefix + "plan_distance/prior_posterior", prior_post_distance.mean().item(), i)

        # make the prior close to the encoder output (vq input). also add the vq posterior loss here.
        return beta_post * posterior_losses["loss"] + beta_pr * prior_losses["loss"]

    return vq_plan_fn


def get_parse_interaction_from_episode_fn(get_contact_fn, window=3, min_contact_len=1, max_contact_len=80,
                                          max_interaction_len=0, bounds=False):
    """
    Parse contact
    Parameters
    ----------
    get_contact_fn
    window
    min_contact_len
    max_contact_len
    max_interaction_len
    bounds

    Returns
    -------

    """
    W = 4 if bounds else 2

    def parse_interaction_from_episode_fn(inputs, outputs):
        # first mask by contact
        contact = get_contact_fn(inputs)

        if len(contact) <= min_contact_len:
            # logger.warn("Skipping episode -- short episode!")
            return np.zeros((0, W)), 0

        # first pass: smooth
        mask = np.zeros(contact.shape, dtype=bool)
        for i in range(len(mask)):
            start = max(i - window, 0)
            end = min(i + window, len(mask))
            mask[i] = np.max(contact[start:end])
        # second pass: chunk
        # print(mask.shape)
        mask_pre = np.concatenate([[False], mask[:-1]])
        # first idx with smoothed contact
        contact_begins = np.logical_and(~mask_pre, mask).nonzero()[0]
        # first idx with no smoothed contact
        contact_ends = np.logical_and(mask_pre, ~mask).nonzero()[0]

        n_skip = 0
        to_keep = None
        if len(contact_begins) > 0:
            # cannot end more than beginnings.
            assert len(contact_ends) in [len(contact_begins) - 1,
                                         len(contact_begins)], "Too many/few contact endings"
            # no ending, create an artificial ending if we end in contact
            if len(contact_ends) == len(contact_begins) - 1:
                contact_ends = np.append(contact_ends, len(mask))

            c_lens = contact_ends - contact_begins
            to_keep = c_lens >= min_contact_len
            if max_contact_len > 0:
                to_keep = np.logical_and(to_keep, c_lens <= max_contact_len)

        # lists of start then end of each contact. (N, 2)
        # segments = np.stack([contact_begins, contact_ends], axis=1).reshape(-1, 2)
        if len(contact_begins) == 0:
            # logger.warn("Skipping episode -- no contact interactions!")
            return np.zeros((0, W)), 0
        else:
            # an interaction goes from the last end to the next start

            # NOTE : ends are the starting time (include), starts are the first steps with contact (exclude)
            prev_ends = np.concatenate([[0], contact_ends])[:-1]
            next_starts = np.concatenate([contact_begins, [len(mask)]])[1:]
            cstarts = contact_begins
            cends = contact_ends

            if max_interaction_len > 0 and to_keep is not None:
                to_keep = np.logical_and(to_keep, next_starts - prev_ends + 1 <= max_interaction_len)

            if to_keep is not None:
                n_skip = len(prev_ends)
                prev_ends = prev_ends[to_keep]
                n_skip = n_skip - len(prev_ends)  # how much was removed
                next_starts = next_starts[to_keep]
                if bounds:
                    cstarts = cstarts[to_keep]
                    cends = cends[to_keep]

            if bounds:
                # N x 4
                return np.stack([prev_ends, cstarts, cends, next_starts], axis=-1), n_skip
            else:
                # N x 2, from last end, to next starts.
                return np.stack([prev_ends, next_starts], axis=-1), n_skip

    return parse_interaction_from_episode_fn