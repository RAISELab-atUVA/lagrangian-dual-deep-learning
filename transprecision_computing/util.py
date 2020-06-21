#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import settings
import numpy as np
import pandas as pd

from sklearn import preprocessing


def mae(y_true, y_pred):
    """ Compute Mean Absolute Error

    This function computes MAE on the non log
    error

    Parameters
    ----------
        y_true : list(float)
            True value for
            a given sample of data
        y_pred : list(float)
            Predicted value for
            a given sample of data

    Returns
    -------
        MAE : float
            Mean Absolute Error

    """
    y_pred = np.array([10 ** -y for y in y_pred])
    y_true = np.array([10 ** -y for y in y_true])
    return np.mean(np.abs(y_pred - y_true))


def is_dominant(x, y):
    """ Checks if the configuration x is dominant over y

    Parameters
    ----------
        x : list(float)
            configuration
        y : list(float)
            configuration

    Returns
    -------
        Dominance Truth Value : bool
            True if x is dominant over y, False otherwise

    """
    n = len(x) if isinstance(x, list) else x.shape[0]
    return all([x[i] > y[i] for i in range(n)])


def couples(precision):
    """ Counts number of couples dominant dominated """
    n = len(precision)
    couples = []
    for i in range(n):
        x = np.repeat([precision[i]], n, axis=0)
        dominated_idx = np.where(np.all(x > precision, axis=1))[0]
        couples += [(i, j) for j in list(dominated_idx)]

    return couples


def violated_const(precision, error):
    """ Counts number of violated_const

        if x' is dominant on x'' -> -log10(e(x')) > -log10(e(x''))

    """
    n = len(precision)
    violated_const = [(i, j) for (i, j) in couples(precision) if error[i] < error[j]]

    return violated_const


def duplicates(error):
    """ Computes the number of duplicates in the error predicted, especially,
        sums the number of repetition of the 3 most frequent elements.

        This function is used to check the validity of the results predicted
        by the model. As observed previous experimens, high values in the
        multiplier lead to trivial prediction, i.e. for every instance the prediction
        has often the same outcome
    """
    u, c = np.unique(np.round(error, 5), return_counts=True)
    dup = list(zip(u, c))
    dup.sort(key=lambda x: x[1])
    return sum([dup[-1][1], dup[-2][1], dup[-3][1]]) / len(error)


# this procedure is used to create dataset which have a high ratio of constraint violations.
# this allows for consistent value of the regularizator at training time.
def build_dataset(benchmark, n_data, violations_ratio, seed):
    """ Builds a dataset with the desired amount of violated constraints """
    np.random.seed(seed)
    nerr = 30

    suff_label_target = 'err_ds_'
    n_violated_const = int(n_data * violations_ratio)
    labels_target = [suff_label_target + str(i) for i in range(nerr)]

    # reading dataset from csv
    data_file = 'exp_results_{}.csv'.format(benchmark)
    df = pd.read_csv(settings.path_to_repo + 'data/' + data_file, sep=';')
    n_var = len(list(df.filter(regex='var_*')))  # number of variable in the configuration
    # error is capped at 0.95
    for _label_target in labels_target:
        df[_label_target] = [0.95 if x > 0.95 else x for x in df[_label_target]]
        df[_label_target] = [sys.float_info.min if 0 == x else -np.log10(x) for x in df[_label_target]]
    # preprocessing
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(df.iloc[:, 0:n_var])
    y = np.mean(scaler.fit_transform(df[labels_target]), axis=1, dtype='float32').reshape((-1, 1))
    # inject constraint violations
    dataset_violated_const = violated_const(X, y)
    if len(dataset_violated_const) < n_violated_const:
        # the number of violated constraints we want to inject in the dataset exceed the number
        # of violated constraints available in the dataset
        raise ValueError('The desired number of injected constrait violations is not available')
    idx = set()  # set containing indexes of samples in the final dataset used for the training
    idx_control = set()  # set used to count the number of violated constraints used
    while len(idx) != n_violated_const and len(idx_control) < len(dataset_violated_const):
        k = np.random.randint(len(dataset_violated_const))
        idx_control.add(k)
        (i, j) = dataset_violated_const[k]
        idx.add(i)
        idx.add(j)
    while len(idx) < n_data:
        idx.add(np.random.randint(len(X)))
    idx = list(idx)
    return X[idx], y[idx]

