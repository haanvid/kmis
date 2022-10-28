import os
import pandas as pd
import autograd.numpy as np
import time
import util
from nn_risk_model import NNRiskModel

import disc_ope
from sklearn import preprocessing
import joblib


def gaussian_kernel(u):
    return np.exp(-0.5 * np.inner(u,u)) / (np.power(2 * np.pi, len(u)/2.0))


def off_policy_evaluation(h_samp, kernel, behav_pol_val_samp,  states_samp, behav_acts_samp, behav_risks_samp, targ_acts_samp, clip_ratio):
    # Uses no metric. Only learns bandwidth for relaxing the indicator function to a kernel
    n_samp = states_samp.shape[0]
    sum_samp_est = 0
    relaxed_IS_rate = np.zeros(n_samp)

    for i in np.arange(n_samp):
        behav_pol_dens = behav_pol_val_samp[i]
        relaxed_IS_rate[i] = float( kernel((targ_acts_samp[i] - behav_acts_samp[i]) / h_samp) / max(behav_pol_dens, clip_ratio) )
        sum_samp_est +=  relaxed_IS_rate[i] * behav_risks_samp[i][0]

    norm_sum = np.sum(relaxed_IS_rate)

    return sum_samp_est/(norm_sum*1.0)


def metric_evaluation(h_samp, L_samp, kernel, behav_pol_val_samp, states_samp, behav_acts_samp, behav_risks_samp, targ_acts_samp, clip_ratio):

    n_samp = states_samp.shape[0]

    loss = 0
    relaxed_IS_rate = np.zeros(n_samp)
    for i in np.arange(n_samp):

        behav_pol_dens = behav_pol_val_samp[i]
        trans_t_minus_tau = np.matmul(np.transpose(L_samp[i]), (targ_acts_samp[i] - behav_acts_samp[i]))
        relaxed_IS_rate[i] = float(kernel(trans_t_minus_tau / h_samp) / max(behav_pol_dens, clip_ratio))
        loss += relaxed_IS_rate[i] * behav_risks_samp[i][0]
    norm_sum = np.sum(relaxed_IS_rate)

    return loss/(norm_sum*1.0)


def get_L(risk_hess_integrals, n_samp, act_dim):

    start_time = time.time()
    # check if the hessian matrix is symmetric
    def check_symmetric(a, rtol=1e-05, atol=1e-5):
        return np.allclose(a, np.transpose(a,(0,2,1)), rtol=rtol, atol=atol)

    def get_metric(B, EPS=10 ** (-8), reg_multiplier=-2.0):

        B = (B + B.T)/2.0
        B_eigval, B_eigvec = np.linalg.eig(B)
        act_dim = B.shape[0]

        # regularizer
        max_B_eigval = np.max(np.abs(B_eigval))
        gamma = max_B_eigval * 10 ** (reg_multiplier)

        sort_idx = np.abs(B_eigval).argsort()[::-1]
        B_eigval = B_eigval[sort_idx]
        B_eigvec = B_eigvec[:, sort_idx]

        pos_B_eigval = B_eigval[B_eigval > EPS]
        pos_B_eigvec = B_eigvec[:, B_eigval > EPS]
        d_pos = pos_B_eigval.shape[0]

        neg_B_eigval = B_eigval[B_eigval < -EPS]
        neg_B_eigvec = B_eigvec[:, B_eigval < -EPS]
        d_neg = neg_B_eigval.shape[0]

        zero_B_eigval = B_eigval[np.logical_and(B_eigval >= -EPS, B_eigval <= EPS)]
        zero_B_eigvec = B_eigvec[:, np.logical_and(B_eigval >= -EPS, B_eigval <= EPS)]

        sqrt_A_eigval = np.concatenate((np.sqrt(d_pos * pos_B_eigval + gamma), np.sqrt(-d_neg * neg_B_eigval + gamma),
                                        np.sqrt(np.zeros(zero_B_eigval.shape)+ gamma)))
        B_eigvec = np.concatenate((pos_B_eigvec, neg_B_eigvec, zero_B_eigvec), axis=-1)
        # transform matrix L (A = L @ L.T)
        L = B_eigvec @ np.diag(sqrt_A_eigval)
        L = L / (np.linalg.det(L @ L.T) ** (1 / (2 * act_dim)))
        assert np.linalg.det(L @ L.T) > (1.0 - 10**(-4)) and np.linalg.det(L @ L.T) < (1.0 +  10**(-4))

        return L

    assert check_symmetric(risk_hess_integrals) is True
    L = np.zeros([n_samp, act_dim, act_dim])

    for i, B in enumerate(risk_hess_integrals):
        L[i] = get_metric(B)

    return L

def get_h(risk_hess_integrals, risk_second_moments, behav_pol_val_samp, targ_acts_samp):

    n_samp = targ_acts_samp.shape[0]
    act_dim = targ_acts_samp.shape[1]
    R_K = (1.0 / (4 * np.pi)) ** (act_dim / 2.0)
    behav_pol_dens_vals = behav_pol_val_samp
    C_var = R_K * np.mean(np.squeeze(risk_second_moments) / behav_pol_dens_vals)
    risk_hess_tr_integrals = np.sum(risk_hess_integrals[:, np.arange(act_dim), np.arange(act_dim)], axis=1)
    kappa_two = 1.0
    C_bias = (kappa_two * 0.5 * np.mean(risk_hess_tr_integrals)) ** 2
    h = np.power((act_dim * C_var) / (4.0 * n_samp * C_bias), 1.0 / (act_dim + 4.0))
    print('h: {}'.format(h))

    return h


def get_kernel_h_L(states_samp, behav_acts_samp, targ_acts_samp, behav_pol_val_samp, gauss_gen_model):

    n_samp = states_samp.shape[0]
    act_dim = behav_acts_samp.shape[1]
    risk_hess_integrals = gauss_gen_model.get_mean_hess(states_samp, targ_acts_samp)
    risk_second_moments = gauss_gen_model.get_2nd_moment_of_r(states_samp, targ_acts_samp)
    L = get_L(risk_hess_integrals, n_samp, act_dim)
    h = get_h(risk_hess_integrals, risk_second_moments, behav_pol_val_samp, targ_acts_samp)

    return h, L


def direct_method_gen(gauss_model, states_samp, targ_acts_samp):
    mean, std  = gauss_model.get_mean_std(states_samp, targ_acts_samp)
    targ_pol_val = np.mean(mean)
    return targ_pol_val


def slope_evaluation(hs, kernel, behav_pol_val_samp, states_samp, behav_acts_samp, behav_risks_samp, targ_acts_samp, clip_ratio):

    means = []
    widths = []
    IS_rate_means = []

    def get_mean_var(h, kernel, behav_pol_val_samp, states_samp, behav_acts_samp, behav_risks_samp,
                              targ_acts_samp, clip_ratio):
        n_samp = states_samp.shape[0]
        sum_samp_est = 0
        relaxed_IS_rate = np.zeros(n_samp)
        samp_est = np.zeros(n_samp)
        act_dim = behav_acts_samp.shape[1]

        for i in np.arange(n_samp):
            behav_pol_dens = behav_pol_val_samp[i]
            relaxed_IS_rate[i] = float(h**(-act_dim)*kernel((targ_acts_samp[i] - behav_acts_samp[i]) / h) / max(behav_pol_dens, clip_ratio))
            samp_est[i] = relaxed_IS_rate[i] * behav_risks_samp[i][0]
        IS_rate_mean = np.mean(relaxed_IS_rate)

        mean = np.mean(samp_est)
        var = np.mean((samp_est - mean) ** 2) / (n_samp - 1)

        return mean, var, IS_rate_mean

    for h in hs:
        mean, var, IS_rate_mean = get_mean_var(h, kernel, behav_pol_val_samp, states_samp, behav_acts_samp, behav_risks_samp,
                              targ_acts_samp, clip_ratio)
        means.append(mean)
        widths.append(np.sqrt(var))
        IS_rate_means.append(IS_rate_mean)
    intervals = []
    for i in range(len(hs)):
        if i < len(hs) - 1:
            width = max(widths[i], max(widths[i + 1:]))
        else:
            width = widths[i]
        intervals.append((means[i] - 2 * width, means[i] + 2 * width))
    index = 0
    curr = [intervals[0][0], intervals[0][1]]
    for i in range(len(intervals)):
        if intervals[i][0] > curr[1] or intervals[i][1] < curr[0]:
            ### Current interval is not overlapping with previous ones, return previous index
            break
        else:
            ### Take intersection
            curr[0] = max(curr[0], intervals[i][0])
            curr[1] = min(curr[1], intervals[i][1])
            index = i

    return means[index] / IS_rate_means[index], hs[index]


def get_pol_val_est(n_0, n_max, n_spacing, kernel, clip_ratio, trial, learn_metric_only, h_given, config,
            num_bins, states_raw, behav_acts_raw, targ_acts_raw, behav_risks_raw, bias, std, is_gaussian,
            hs=None, gt_policy_val=None, save_reward_nn=False, reward_save_dir = None, batch_size=256):

    result_dir = config['result_dir']
    result_path = result_dir + '/result_{}.csv'.format(trial)
    reward_nn_save_path = reward_save_dir + '/reward_nn'

    i_start = 0
    if os.path.isfile(result_path):
        result_arr = pd.read_csv(result_path).values
        if result_arr.shape[0] != result_arr[-1,0]+1:
            os.remove(result_path)
            i_start = 0
        else:
            i_start = result_arr[-1,0]+1
    else:
        i_start=0

    i_start = int(i_start)
    loop_arr = np.linspace(n_0, n_max, n_spacing)
    loop_arr = loop_arr[i_start:]


    for i, n_sub in enumerate(loop_arr):

        i = i + i_start
        n_samp = int(np.floor(n_sub))
        # disc ope
        states_raw_samp  = states_raw[:n_samp,:]
        behav_acts_raw_samp = behav_acts_raw[:n_samp,:]
        targ_acts_raw_samp = targ_acts_raw[:n_samp, :]
        behav_risks_raw_samp  = behav_risks_raw[:n_samp, :]
        n_samp = states_raw_samp.shape[0] # for the case when actual datasize may be smaller than n_samp
        # standardize the data
        states_scaler = preprocessing.StandardScaler().fit(states_raw_samp)
        states_samp = states_scaler.transform(states_raw_samp)

        actions_scaler = preprocessing.StandardScaler().fit(behav_acts_raw_samp)
        behav_acts_samp = actions_scaler.transform(behav_acts_raw_samp)
        targ_acts_samp  = actions_scaler.transform(targ_acts_raw_samp)

        rewards_scaler = preprocessing.StandardScaler().fit(behav_risks_raw_samp)
        behav_risks_samp = rewards_scaler.transform(behav_risks_raw_samp)

        behav_pol_val_samp = np.ones(n_samp) / ((2.0 * config['act_bound']) ** config['act_dim']) * np.prod(actions_scaler.scale_)
        behav_pol_val_targ_acts_samp = behav_pol_val_samp
        # Learn reward regressor and get h and L from it
        gauss_gen_model = NNRiskModel(states_samp.shape[-1], behav_acts_samp.shape[-1], config['hidden'], config['l2'], config['lr'], use_l2=config['use_l2'], dropout_rate=config['dropout_rate'], use_dropout = config['use_dropout'])
        if learn_metric_only:
            gauss_gen_model.load(filepath = reward_nn_save_path)
        else:
            gauss_gen_model.train(states_samp, behav_acts_samp, behav_risks_samp, batch_size)
        if save_reward_nn and i == n_spacing-1 :
            os.makedirs(reward_save_dir, exist_ok=True)
            joblib.dump(states_scaler, reward_save_dir + '/states_scaler.joblib')
            joblib.dump(actions_scaler, reward_save_dir + '/actions_scaler.joblib')
            joblib.dump(rewards_scaler, reward_save_dir + '/rewards_scaler.joblib')
            gauss_gen_model.save(reward_nn_save_path)

        h_samp, L_samp = get_kernel_h_L(states_samp, behav_acts_samp, targ_acts_samp, behav_pol_val_targ_acts_samp, gauss_gen_model)

        # Compute target policy value estimates
        slope_se=0 ; slope_eval =0
        metric_slope_se=0 ; metric_slope_eval =0
        dm_gen_se =0; dm_gen_eval=0
        disc_se = 0; disc_eval = 0
        h_slope = 0

        if learn_metric_only:
            metric_eval = metric_evaluation(h_given, L_samp, kernel, behav_pol_val_samp, states_samp,
                                                 behav_acts_samp, behav_risks_samp, targ_acts_samp, clip_ratio)
            off_pol_eval = off_policy_evaluation(h_given, kernel, behav_pol_val_samp, states_samp, behav_acts_samp,
                                                     behav_risks_samp, targ_acts_samp, clip_ratio)
        else:
            metric_eval = metric_evaluation(h_samp, L_samp, kernel, behav_pol_val_samp, states_samp, behav_acts_samp,
                                                behav_risks_samp, targ_acts_samp, clip_ratio)
            slope_eval, h_slope = slope_evaluation(hs, kernel, behav_pol_val_samp, states_samp, behav_acts_samp,
                                            behav_risks_samp, targ_acts_samp, clip_ratio)
            metric_slope_eval = metric_evaluation(h_slope, L_samp, kernel, behav_pol_val_samp, states_samp,
                                                 behav_acts_samp, behav_risks_samp, targ_acts_samp, clip_ratio)
            off_pol_eval = off_policy_evaluation(h_samp, kernel, behav_pol_val_samp, states_samp, behav_acts_samp,
                                                     behav_risks_samp, targ_acts_samp, clip_ratio)
            dm_gen_eval = direct_method_gen(gauss_gen_model, states_samp, targ_acts_samp)

            EPS = 1e-8
            a_max = behav_acts_raw_samp.max(axis=0) + EPS
            a_min = behav_acts_raw_samp.min(axis=0)


            disc_eval = disc_ope.disc_evaluation(num_bins, states_raw_samp, behav_acts_raw_samp, behav_risks_raw_samp.reshape(-1),
                                     targ_acts_raw_samp, clip_ratio, a_min, a_max, is_gaussian, bias, std)

            if rewards_scaler is not None:
                slope_eval        = rewards_scaler.inverse_transform(np.array([slope_eval]))[0]
                metric_slope_eval = rewards_scaler.inverse_transform(np.array([metric_slope_eval]))[0]
                dm_gen_eval       = rewards_scaler.inverse_transform(np.array([dm_gen_eval]))[0]

            slope_se   = (slope_eval - gt_policy_val) ** 2
            metric_slope_se = (metric_slope_eval - gt_policy_val) ** 2
            dm_gen_se  = (dm_gen_eval-gt_policy_val)**2
            disc_se = (disc_eval-gt_policy_val)**2

        if rewards_scaler is not None:
            metric_eval = rewards_scaler.inverse_transform(np.array([metric_eval]))[0]
            off_pol_eval = rewards_scaler.inverse_transform(np.array([off_pol_eval]))[0]
        metric_se  = (metric_eval  - gt_policy_val)**2
        off_pol_se = (off_pol_eval - gt_policy_val)**2

        # n_samp: datasize used for estimation
        # h_samp: bandwidth learned with Kallus and Zhou's estimator
        # h_slope: bandwidth selected by SLOPE
        # slope_eval: evaluation result of SLOPE
        # metric_slope_eval: evaluation result of SLOPE + KMIS
        # off_pol_eval: evaluation result of Kallus and Zhou's estimator
        # metric_eval: evaluation result of Kallus and Zhou + KMIS
        # dm_gen_eval: evaluation result of DM
        # disc_eval: evaluation result of discrete OPE
        util.write_csv(i, n_samp, h_samp, h_slope, metric_slope_eval, metric_eval, slope_eval, off_pol_eval, dm_gen_eval, metric_slope_se, metric_se, slope_se, off_pol_se, dm_gen_se, disc_se, disc_eval,
                        result_path)

    return
