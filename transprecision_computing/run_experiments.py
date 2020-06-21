
from dataset import *
from Regressor import   *
from SBRregressor import  *
from SBRregressor2 import  *
from settings import *
from util import *
import argparse

import numpy as np, pandas as pd
import  sys, copy, pickle

if False:
    path_to_repo = '/content/drive/My Drive/Transprecision_Computing/IJCAI/'
    from google.colab import drive
    drive.mount('/content/drive')
sys.path.insert(1,'/content/drive/My Drive/Transprecision_Computing/IJCAI/')

import torch
Ten = torch.FloatTensor
iTen = torch.LongTensor



def test(benchmark, violated_const_ratio, test_seed, start_point_seed = 2):



    res = {'test_seed':test_seed} # store all model results
    params = {'epochs': 150,
                   'n_data': 4000,
                   'batch_size': 256,
                   'violated_const_ratio': violated_const_ratio,  # this is used to create a trainig set with a specific
                   'benchmark': benchmark,
                   'split': [0.5, 0.25, 0.25],
                   'seed': test_seed}

    d_trainall = Dataset(params, 'train', 'cpu')
    d_test = Dataset(params, 'test', 'cpu')
    d_valall = Dataset(params, 'valid', 'cpu')
    res['d_test'] = d_test
    X_test, y_test = d_test._dataset

    val_size = 300 # fix validation set size
    for train_size in [200, 400, 600, 800, 1000]:
        res[train_size] = {}
        for split_seed in range(20):
            print('train size = {} , seed = {}'.format(train_size, split_seed))
            np.random.seed(split_seed)
            X_train,y_train  = copy.deepcopy( d_trainall._dataset)
            idx_train = np.random.choice( list(range(len(X_train))), train_size)
            X_train, y_train = X_train[idx_train,:], y_train[idx_train]

            d_train = copy.deepcopy(d_trainall)
            d_train._dataset = (X_train, y_train)
            y_med_pred = np.median(y_train)* np.ones(len(y_test))

            X_val, y_val = copy.deepcopy(d_valall._dataset)
            idx_val = np.random.choice(list(range(len(X_val))), val_size)
            X_val, y_val = X_val[idx_val,:], y_val[idx_val]
            d_val = copy.deepcopy(d_valall)
            d_val._dataset = (X_val, y_val)
            res[train_size][split_seed] ={'d_val':d_val, 'd_train': d_train}


            model_1 = Regressor(params, d_train, d_test, d_val,  start_point_seed) # start_point_seed is random seed of pytorch
            # make sure all models starting from the same initial points
            model_1.train()
            tmp = model_1.test()
            res[train_size][split_seed]['model_1_perf'] = copy.deepcopy(tmp)


            # regularization with single multiplier =1
            model_2_1 = SBRregressor(params, d_train, d_test, d_val,  start_point_seed)
            model_2_1.train(options = {'mult_fixed':True})
            tmp = model_2_1.test()
            res[train_size][split_seed]['model_2_1_perf'] = copy.deepcopy(tmp)



            ###################regularization with single multiplier updated gradually, starts with 0

            model_2 = SBRregressor(params, d_train, d_test, d_val,  start_point_seed)
            if split_seed == 0:
                model_2.opt_lr_rate()
                best_lr_model_2 = copy.deepcopy( model_2._LR_rate)
            else:
                model_2._LR_rate = copy.deepcopy( best_lr_model_2)
                model_2.train(options={'mult_fixed': False})
            tmp = model_2.test()
            res[train_size][split_seed]['model_2_perf'] = copy.deepcopy(tmp)


            ########## regularization with a multiplier for each constraint, each multiplier has updated gradually, starts with 0


            model_3 = SBRregressor2(params, d_train, d_test, d_val,  start_point_seed)
            if split_seed == 0:
                model_3.opt_lr_rate()
                best_lr_model_3 = copy.deepcopy(model_3._LR_rate)
            else:
                model_3._LR_rate = copy.deepcopy( best_lr_model_3)
                model_3.train(options={'mult_fixed': False})

            tmp = model_3.test()
            res[train_size][split_seed]['model_3_perf'] = copy.deepcopy(tmp)


            res[train_size][split_seed]['model_3'] = copy.deepcopy(model_3)
            res[train_size][split_seed]['model_2'] = copy.deepcopy( model_2)
            res[train_size][split_seed]['model_1'] = copy.deepcopy( model_1)
            res[train_size][split_seed]['model_2_1'] = copy.deepcopy(model_2_1)
            res[train_size][split_seed]['dump_model_perf'] = [copy.deepcopy(mae(y_test, y_med_pred)), 0, 0]

    filename = str(benchmark) + '_test_seed_{}'.format(test_seed)+\
                    "_vconst" + str(violated_const_ratio) + '_start_point_seed_'+ str( start_point_seed) +'.pkl'

    file_handle = open(path_to_repo + 'results/' +filename, 'wb')
    pickle.dump(res, file_handle)



#test('correlation', 0.6, 2, 1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='KWB Parser')

    parser.add_argument('--benchmark', type=str, help='benchmark: <correlation> or <convolution>')
    parser.add_argument('--violated_const_ratio', default=0.5, type=float,
                        help='ratio of violated constraint in training set')
    parser.add_argument('--test_seed', default=0, type=int,
                        help='random_seed to generate test set')

    parser.add_argument('--start_point_seed', default=2, type=int,
                        help='random seed for initial points in deep model')

    args = parser.parse_args()


    test(args.benchmark,
         args.violated_const_ratio,
         args.test_seed,
         args.start_point_seed)