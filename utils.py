import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import sklearn

import copy, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from sklearn.metrics import *

torch.set_default_tensor_type(torch.cuda.FloatTensor)


DATA_PATH = '/home/cutran/research/privacy_with_fairness/datasets/'

### Evaluation Metrics 
def compute_fairness_score(y_true, y_pred, z_true):
  acc = accuracy_score(y_true, y_pred)
  p1 = y_pred[z_true==1]
  p0 = y_pred[z_true==0]
  p1 = np.sum(p1)/float(len(p1))
  p0 = np.sum(p0)/float(len(p0))


def load_dataset(name, rm_pfeat=False, classes=[0, 1], to_numeric=True):
    # name = params.dataset
    # pfeat = params['pfeat']
    if name == 'census':
        dataset = pd.read_csv(DATA_PATH + 'census.csv', na_values='?', skipinitialspace=True)
        x_feat_c = ['workclass', 'marital-status', 'occupation', 'relationship', 'race',
                    'native-country']
        x_feat_n = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        p_feat = 'sex'
        y_feat = 'fnlwgt'
        del dataset['education']
    elif name == 'default':
        dataset = pd.read_csv(DATA_PATH + 'default.csv', na_values='?', skipinitialspace=True)
        x_feat_c = ['education', 'marriage', 'bill_amt1', 'bill_amt2', 'bill_amt3',
                    'bill_amt4', 'bill_amt5', 'bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3',
                    'pay_amt4', 'pay_amt5', 'pay_amt6']
        x_feat_n = ['limit_bal', 'age', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
        p_feat = 'sex'
        y_feat = 'default'

    elif name == 'compas':
        dataset = pd.read_csv(DATA_PATH + 'compas.csv', na_values='', skipinitialspace=True)
        x_feat_c = ['sex', 'c_charge_degree', 'c_charge_desc']
        x_feat_n = ['age', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count',
                    'priors_count', 'decile_score', 'is_recid', 'priors_count']
        p_feat = 'race'
        y_feat = 'two_year_recid'
    x_feat = [p_feat] + x_feat_n + x_feat_c

    if rm_pfeat:
        del dataset[p_feat]
        del x_feat[x_feat.index(p_feat)]

    dataset = dataset.dropna().reset_index(drop=True)
    # Data Preprocessing
    if to_numeric:
        lb_make = sklearn.preprocessing.LabelEncoder()
        obj_df = dataset.copy()
        for feat in x_feat_c + [p_feat]:  # list(obj_df.columns):
            dataset.loc[:, feat] = lb_make.fit_transform(obj_df[feat])

    # Set target to [-1, 1]
    if name in ['census', 'default', 'compas']:
        assert len(classes) == 2
        a, b = min(dataset[y_feat]), max(dataset[y_feat])
        dataset[y_feat] = dataset[y_feat].replace(a, classes[0])
        dataset[y_feat] = dataset[y_feat].replace(b, classes[1])

    # move target values (y) in the first column
    targetcol = dataset[y_feat]
    dataset.drop(labels=[y_feat], axis=1, inplace=True)
    dataset.insert(0, y_feat, targetcol)

    return dataset, x_feat_c, x_feat_n, y_feat, p_feat




def load_data(file_name):
    if file_name!='bank':
        dataset, x_feat_c, x_feat_n, y_feat, p_feat = load_dataset(file_name)
        pd00 = copy.deepcopy(dataset)

        label_name, z_name = y_feat, p_feat
        feats = [col for col in x_feat_c + x_feat_n if col not in [label_name, z_name]]
        if file_name == 'compas':
            pd00 = pd00[pd00[z_name].isin([0, 2])]
            pd00.loc[pd00[z_name] == 2, z_name] = 1
    else:
        pd00, feats = load_bank_dataset()
        z_name = 'z'
        label_name = 'label'


    assert (pd00[z_name].min() == 0) and (pd00[z_name].max() == 1)
    assert (pd00[label_name].min() == 0) and (pd00[label_name].max() == 1)

    return pd00, label_name, z_name, feats


def get_data_loader(pd00, feats, label_name, z_name, seed=0):
    X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(pd00[feats].values, pd00[label_name].values, pd00[z_name].values,
                                                                         test_size=0.2,
                                                                         stratify=pd00[label_name].values, random_state=seed)

    X_train, X_val, y_train, y_val, Z_train, Z_val = train_test_split(X_train, y_train, Z_train,
                                                                         test_size= 0.25,
                                                                         stratify= y_train,
                                                                         random_state= seed)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test =  scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    return X_train, X_val, X_test, y_train, y_val, y_test, Z_train, Z_val, Z_test
