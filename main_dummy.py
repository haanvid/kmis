import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import autograd.numpy as np
from off_pol_eval_functs import *
import time
import tensorflow as tf
import random
import pickle
import pandas as pd
from sklearn import preprocessing
import joblib

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def gen_unif_state(n_samp, state_dim, act_bound=1.0):
    states = np.array(np.random.uniform(-act_bound, act_bound, (n_samp, state_dim)))
    return states


if __name__ == "__main__":

    def boolean(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(description='Trial, Random Seed (scipy, numpy, sklearn)')
    parser.add_argument("--n_0", help="starting number of samples", default=300, type=int)
    parser.add_argument("--n_spacing", help="number of spacing between n_0 and n_max", default=21, type=int) #21
    parser.add_argument("--n_max", help="max data size", default=40000, type=int) # 40000
    parser.add_argument("--num_bins", help="descritization bins", default=10, type=int)

    parser.add_argument('--result_dir',       type=str, default='dummy_result', help="file path")
    parser.add_argument('--sampled_data_dir', type=str, default='dummy_result', help="batch data path")
    parser.add_argument('--env_name', default='dummy', type=str)
    parser.add_argument("--save_batch_data", help="True to save data", default=True, type=boolean)

    parser.add_argument('--is_gaussian', type=boolean, default=False, help="True to use guassian otherwise uniform")
    parser.add_argument('--state_dim', type=int, default=2, help="int bigger than 2")
    parser.add_argument('--act_dim', type=int, default=2,
                        help="dimension of action. 1 can be handled with proper behavior policy defined")
    parser.add_argument('--clip_ratio', type=float, default=0.0, help="behavior policy pdf value min clip")
    parser.add_argument('--risk_std', type=float, default=0.0, help="risk_std")

    parser.add_argument("--learn_metric_only", help="True to use given h", default=False, type=boolean)
    parser.add_argument("--given_h", help="given kernel bandwidth", default=0.1, type=float)
    parser.add_argument("--gpu_assign", help="idx of gpu assigned", default=-1, type=int)

    parser.add_argument("--pid", help="pid for allocating jobs", default=0, type=int)  # 1000
    parser.add_argument('--trial', dest='trial', action='store', type=int, default=0, help="trial")

    parser.add_argument('--batch_size', dest='batch_size', action='store', type=int, default=256, help="batch_size")
    parser.add_argument("--save_reward_nn", help="True to save", default=True, type=boolean)

    parser.add_argument("--hidden", help="hidden size of NNriskmodel", default=128, type=int)
    parser.add_argument("--use_l2", help="False for not using", default=False, type=boolean)
    parser.add_argument("--use_dropout", help="False for not using", default=True, type=boolean)
    parser.add_argument("--l2", help="l2 regularization coeff for NNriskmodel", default=0.0, type=float)
    parser.add_argument("--dropout_rate", help="dropout rate", default=0.5, type=float)

    parser.add_argument("--lr", help="hidden size of NNriskmodel", default=0.0005, type=float)
    parser.add_argument("--standardize_data", help="True to standardize", default=True, type=boolean)
    parser.add_argument("--act_bound", help="action bound", default=1.0, type=float)

    start_time = time.time()
    args = parser.parse_args()
    config = vars(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_assign'])
    result_dir = config['result_dir']
    sampled_data_dir = config['sampled_data_dir']
    learn_metric_only = config['learn_metric_only']
    trial = config['trial']
    given_h = config['given_h']
    clip_ratio = config['clip_ratio']
    hidden = config['hidden']
    l2 = config['l2']
    lr = config['lr']
    n_0 = config['n_0']
    n_spacing = config['n_spacing']
    n_max = config['n_max']
    state_dim = config['state_dim']
    act_dim = config['act_dim']
    batch_size = config['batch_size']
    save_reward_nn = config['save_reward_nn']
    pid = config['pid']
    standardize_data = config['standardize_data']
    risk_cond_density = 'gaussian'
    initial_state_density = 'uniform'
    is_gaussian = config['is_gaussian']
    num_bins = config['num_bins']
    risk_std = config['risk_std']
    act_bound = config['act_bound']
    ####################################################################################################################
    hs = [x for x in np.logspace(1, 7, num=7, base=0.5)]
    hs.reverse()
    # ground truth policy value
    gt_policy_val = 0.0
    ##################### grid search ##################################################################################
    trial_list = [i for i in range(100)]  # 100
    trial = trial_list[pid % len(trial_list)]
    config.update({'trial': trial})
    ################# Fix random seed ##################################################################################
    np.random.seed(trial)  # Will be using trial index as the random seed
    tf.set_random_seed(trial)  # fix TF random seed
    random.seed(trial)
    ####################################################################################################################
    # exp naming
    exp_name = ""
    if config['learn_metric_only']:
        exp_name = "given_h%.5f" % (config['given_h'])
    else:
        exp_name = "learned_h"
    exp_name = exp_name + '/hidden_%d'% (hidden)
    if config['batch_size'] != 256:
        exp_name = exp_name + '/batch_size_%d'% (batch_size)
    if config['use_l2']:
        exp_name = exp_name + '/l2_%.5f'% (l2)
    if config['use_dropout']:
        exp_name = exp_name + '/dropout_%.5f' % (config['dropout_rate'])
    exp_name = exp_name + '/lr_%.5f' % (lr)
    exp_name = exp_name + '/clip_ratio_{}'.format(np.format_float_scientific(clip_ratio, precision=1, exp_digits=2))
    exp_name = exp_name + '/num_bins_%d' % (config['num_bins'])
    exp_name = exp_name + '/risk_std_%.1f' % (config['risk_std'])
    ####################################################################################################################
    # Data saving and loading
    result_dir = os.path.join(result_dir, exp_name)
    config.update({'result_dir': result_dir})
    os.makedirs(result_dir, exist_ok=True)
    start_time = time.time()
    batch_data_path = os.path.join(sampled_data_dir, 'trial_{}.pkl'.format(trial))

    if os.path.exists(batch_data_path):
        print("Loading existing batch data from : {}".format(batch_data_path))
        batch_data_dict = load_dict(batch_data_path)
        assert batch_data_dict['act_dim'] == act_dim and batch_data_dict['state_dim'] == state_dim
        states = batch_data_dict['states']
        targ_acts = batch_data_dict['targ_acts']
        behav_acts = batch_data_dict['behav_acts']
        behav_risks = batch_data_dict['behav_risks']
    else:
        states     = gen_unif_state(n_max,state_dim, act_bound)
        behav_acts = gen_unif_state(n_max,state_dim, act_bound)
        lin_pol = 0.5
        targ_acts = lin_pol*states
        behav_risks =  -np.abs(lin_pol*states[:,0]-behav_acts[:,0]) + (risk_std * np.random.randn(targ_acts.shape[0]) )
        behav_risks = behav_risks.reshape([-1,1])

        if args.save_batch_data:
            os.makedirs(sampled_data_dir, exist_ok=True)
            batch_data_dict = {"act_dim":act_dim, "state_dim":state_dim, 'states':states, 'targ_acts':targ_acts,
                               'behav_acts':behav_acts, 'behav_risks':behav_risks}
            save_dict(batch_data_dict, batch_data_path)
    ####################################################################################################################
    start_time = time.time()
    kernel = gaussian_kernel
    reward_save_dir = result_dir + '/reward_%d'%(trial)
    os.makedirs(reward_save_dir, exist_ok=True)
    # bias and std not used
    bias=0.0
    std = 0.0

    get_pol_val_est(n_0, n_max, n_spacing, kernel, clip_ratio, trial, learn_metric_only, given_h, config,
                    num_bins, states, behav_acts, targ_acts, behav_risks, bias, std, is_gaussian,
                    hs= hs, gt_policy_val=gt_policy_val, save_reward_nn=save_reward_nn, reward_save_dir = reward_save_dir, batch_size=batch_size)

    print("Finish! ({:.3f})".format(time.time() - start_time))