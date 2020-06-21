""" Strongly inspired by Ferdinando Fioretto's code """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import util

import numpy as np

from torch_inputs import *


class DataIterator(object):
    def __init__(self, dataset, batchsize, device):
        # object reference (a list of tuples of lists)
        self._dataset = dataset
        self._len = len(dataset[0])
        # keep track of current index
        self._index = 0
        # the batch size
        self._batchsize = batchsize
        self._device = device

    def __next__(self):
        ''''Returns the next value from object's lists '''

        n = min(self._len, self._index + self._batchsize)
        if self._index < self._len:
            x = self._dataset[0][self._index:n]
            y = self._dataset[1][self._index:n]
            self._index = n
            return Ten(x).to(self._device), Ten(y).to(self._device)

        # End of Iteration
        raise StopIteration


class Dataset(object):

    def __init__(self, params, mode, device):
        assert mode in ['train', 'test', 'valid']
        np.random.seed(params['seed'])
        #self._const = 0  # constrain counter
        self._device = device
        self._n_data = params['n_data']
        self._benchmark = params['benchmark']
        self._batchsize = params['batch_size']
        self._violated_const_ratio = params['violated_const_ratio'] if mode == 'train' else 0
        # builds ad hoc dataset, the number of violated_ constraints can be tuned
        (X, y) =  util.build_dataset(self._benchmark, self._n_data, self._violated_const_ratio, params['seed'])
        self._n_var = len(X[0])
        indices = self._get_indexes(params, self._n_data, mode, params['seed'])
        X, y = X[indices], y[indices]
        self._dataset = tuple([X, y])

    @property
    def n_var(self):
        return self._n_var

    #@property
    #def const(self):
    #    return self._const

    def _get_indexes(self, params, n_data, mode, seed):
        indices = np.arange(n_data)
        np.random.seed(seed)
        np.random.shuffle(indices)
        split_size = dict()
        modeidx = {'train': 0, 'test': 1, 'valid': 2}
        for m in ['train', 'test', 'valid']:
            split_size[m] = int(params['split'][modeidx[m]] * n_data)
        if mode == 'train':
            indices = indices[0:split_size['train']]
        elif mode == 'test':
            indices = indices[split_size['train']:split_size['test'] + split_size['train']]
        else:
            indices = indices[split_size['train'] + split_size['test']:-1]
        return indices

    def __iter__(self):
        return DataIterator(self._dataset, self._batchsize, self._device)
