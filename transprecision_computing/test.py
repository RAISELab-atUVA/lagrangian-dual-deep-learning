#!/usr/bin/env python
# -*- coding: utf-8 -*-
# USE THIS FOR REFERENCE ONLY
import csv
import settings
import argparse
from dataset import Dataset

import numpy as np
import matplotlib.pyplot as plt

from torch_inputs import *

from SBRregressor2 import SBRregressor2
from SBRregressor import SBRregressor
from Regressor import Regressor


# seeds used for the shuffling of the dataset, different
# seeds will result in different samples extracted for
# the learning process
seeds = [2, 3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
         43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
         101, 103, 107, 109,113, 127, 131, 137, 139, 149,
         151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
         199, 211, 223, 227, 229, 233, 239, 241, 251, 257,
         263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
         317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
         383, 389, 397, 401, 409, 419, 421, 431, 433, 439,
         443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
         503, 509, 521, 523, 541]


def test(n_data, benchmark, violated_const_ratio):

    model_1_mae, model_2_mae, model_3_mae = [], [], []
    model_1_violated_const, model_2_violated_const, model_3_violated_const = [], [], []

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    train_split = 0.5
    test_split = 0.4
    val_split = 0.1

    for seed in seeds:

        params = {'epochs': 150,
                  'n_data': n_data,
                  'batch_size': 256,
                  'violated_const_ratio': violated_const_ratio, # this is used to create a trainig set with a specific
                                                                # amount of contraint violations
                  'benchmark': benchmark,
                  'split': [train_split, test_split, val_split],
                  'seed': seed}

        # the 3 approaches are tested on the same sample of data
        d_train = Dataset(params, 'train', device)
        d_test = Dataset(params, 'test', device)
        d_val = Dataset(params, 'valid', device)

        # regressor
        model_1 = Regressor(params, d_train, d_test, d_val)
        model_1.train()
        tmp = model_1.test()
        model_1_mae.append(tmp[0])
        model_1_violated_const.append(tmp[1])

        # regularization with single multiplier
        model_2 = SBRregressor(params, d_train, d_test, d_val)
        model_2.train()
        tmp = model_2.test()
        model_2_mae.append(tmp[0])
        model_2_violated_const.append(tmp[1])

        # regularization with a multiplier for each constraint
        model_3 = SBRregressor2(params, d_train, d_test, d_val)
        model_3.train()
        tmp = model_3.test()
        model_3_mae.append(tmp[0])
        model_3_violated_const.append(tmp[1])

    mae = list(zip(model_1_mae, model_2_mae, model_3_mae))
    violated_const = list(zip(model_1_violated_const, model_2_violated_const, model_3_violated_const))

    base_filename = str(benchmark) +\
                    "_tr" + str(int(n_data * train_split)) +\
                    "_ts" + str(int(n_data * test_split)) +\
                    "_v" + str(int(n_data * val_split)) +\
                    "_vconst" + str(violated_const_ratio)

    store(base_filename, mae, violated_const)


def store(filename, mae, violated_const):
    n = len(mae)
    with open(settings.path_to_repo + 'results/' + filename + '.csv', mode='a') as f:
        writer = csv.writer(f, delimiter=';')
        for i in range(n):
            row = []
            row += mae[i]
            row += violated_const[i]
            writer.writerow(row)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='KWB Parser')

    parser.add_argument('--n_data', type=int, help='number of data samples')
    parser.add_argument('--benchmark', type=str, help='benchmark: <correlation> or <convolution>')
    parser.add_argument('--violated_const_ratio', default=0.6, type=float,
                        help='ratio of violated constraint in training set')

    args = parser.parse_args()

    test(args.n_data,
         args.benchmark,
         args.violated_const_ratio)
