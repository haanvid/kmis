import os
import autograd
import autograd.numpy as np
import tensorflow as tf
from scipy import spatial
import joblib
import pandas as pd


def batch_hessian(f, x):
    """
    f: (N x D) -> (D)
    x: (N x D)

    :return: Hessian matrices (N x D x D)
    """
    assert len(x.shape) == 2
    D = x.shape[1]
    def g(d):
        def g_inner(x):
            return autograd.elementwise_grad(f)(x)[:, d]
        return g_inner
    H = np.transpose([autograd.elementwise_grad(g(d))(x) for d in range(D)], [1, 0, 2])
    return H

#
def batch_hessian_tf(f, x):
    """
    x: Tensor or placeholder (N x D)
    f: Tensor (N)

    :return: Hessian matrices Tensor (N x D x D)
    """
    assert len(x.shape) == 2 and len(f.shape) == 1
    D = int(x.shape[1])
    grads = tf.gradients(f, x)[0]  # N x D
    result = []
    for d in range(D):
        result.append(tf.gradients(grads[:, d], x)[0])
    result = tf.transpose(result, [1, 0, 2])
    return result


def shuffle_data(states, behav_acts, behav_risks, sample_size):
    idx_arr = np.arange(states.shape[0])
    shuffle_idx = np.random.choice(idx_arr, size= sample_size,replace=False)
    states = states[shuffle_idx]
    behav_acts = behav_acts[shuffle_idx]
    behav_risks = behav_risks[shuffle_idx]

    return states, behav_acts, behav_risks




def write_csv(i, n_samp, h_samp, h_slope, metric_slope_eval, metric_eval, slope_eval, off_pol_eval, dm_gen_eval, metric_slope_se, metric_se, slope_se, off_pol_se, dm_gen_se, disc_se=None, disc_eval=None, csv_path=None):

    data = {
        'i': [i],
        'n_samp': [n_samp],
        'h_samp': [h_samp],
        'h_slope': [h_slope],
        'metric_slope': [metric_slope_eval],
        'metric': [metric_eval],
        'slope': [slope_eval],
        'iso': [off_pol_eval],
        'dm': [dm_gen_eval],
        'metric_slope_se': [metric_slope_se],
        'metric_se':[metric_se],
        'slope_se': [slope_se],
        'off_pol_se':[off_pol_se],
        'dm_gen_se':[dm_gen_se]
    }

    if disc_se != None and disc_eval != None:
        data.update({'disc_se': [disc_se],
                     'disc_eval': [disc_eval]})

    # Make data frame of above data
    df = pd.DataFrame(data)

    # append data frame to CSV file
    if i==0:
        df.to_csv(csv_path, mode='w', index=False, header=True)
    else:
        df.to_csv(csv_path, mode='a', index=False, header=False)

    return

