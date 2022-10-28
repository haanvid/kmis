import numpy as np
from scipy.stats import mvn


def get_behavior_policy_prob(act_dim, low, upp, mean, std, a_min, a_max, is_gaussian):
    # Nomral distribution
    if is_gaussian:
        assert mean.ndim == 1

        cov = std ** 2 * np.eye(act_dim)
        assert upp.shape[0] == act_dim

        gaussian_prob, _ = mvn.mvnun(low, upp, mean, cov)
        return gaussian_prob

    else: # uniform dist
        assert np.prod(a_max - a_min) != 0
        unif_prob = 1. / np.prod(a_max - a_min)
        unif_prob *= np.prod(upp - low)
        return unif_prob


def histogram(action_seq, state_seq, num_bins, with_prob=True, bias=0.2, std=0.5, a_min=-np.ones(2), a_max=np.ones(2), is_gaussian=True):
    # num_bins: number of bins per dimension
    width = np.zeros(a_max.shape)

    start_points_list = []
    end_points_list = []
    for i_dim in range(a_max.shape[0]):
        width[i_dim] = (a_max[i_dim] - a_min[i_dim]) / num_bins
        start_points = np.linspace(a_min[i_dim], a_max[i_dim], num_bins, endpoint=False)
        start_points_list.append(start_points)
        end_points= start_points + width[i_dim]
        end_points_list.append(end_points)

    start_points_arr = np.array(start_points_list)
    end_points_arr = np.array(end_points_list)

    seq_element_index = []
    prob_list = []
    for i in range(len(action_seq)):
        out_of_bound = 0
        state = state_seq[i]
        action = action_seq[i]
        position = []
        for i_dim, x in enumerate(action):

            where = (start_points_arr[i_dim] <= x) * (x < end_points_arr[i_dim])
            if where.any() ==False:
                position.append(np.array([-1]))
            else:
                index = np.nonzero(where)
                position.append(index[0])

        seq_element_index.append(position)
        # CHECK
        behavior_mean = state + bias
        low = a_min.reshape([-1,1]) + np.array(position) * width.reshape([-1,1])
        high = a_min.reshape([-1,1]) + (np.array(position) + 1) * width.reshape([-1,1])
        act_dim = len(action)

        if with_prob: # with probability
            prob = get_behavior_policy_prob(act_dim, low.reshape(-1), high.reshape(-1), mean=behavior_mean.reshape(-1), std=std, a_min=a_min, a_max=a_max, is_gaussian=is_gaussian)  # (1, 1)
            prob_list.append(prob)
    if with_prob:
        return np.array(seq_element_index), np.array(prob_list)
    return np.array(seq_element_index)


def disc_evaluation(num_bins, states, behavior_actions, behavior_risk, target_actions, q_clip_ratio, a_min, a_max, is_gaussian, bias=0.2, std=0.5):

    num_samples, a_dim = behavior_actions.shape
    behavior_position, q_density = histogram(behavior_actions, states, num_bins, bias=bias, std=std,
                                             a_min=a_min, a_max=a_max, is_gaussian = is_gaussian)
    target_position = histogram(target_actions, states, num_bins, with_prob=False, bias=bias, std=std,
                                a_min=a_min, a_max=a_max, is_gaussian = is_gaussian)

    assert np.any(behavior_position != -1)
    EPS = 1e-9

    ratios = (target_position == behavior_position).all(1).squeeze() / np.clip(q_density, a_min = q_clip_ratio, a_max=None).squeeze()
    ratios = ratios + EPS
    value = ratios * behavior_risk / np.sum(ratios)
    return value.sum()

