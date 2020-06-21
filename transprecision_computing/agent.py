""" Strongly inspired by Ferdinando Fioretto's code """


import util, copy
from sklearn.metrics import mean_squared_error as mse
from torch_inputs import *
import numpy as np
import network

class AbstractAgent():
    def __init__(self, params, d_train, d_test, d_val, start_point_seed=0):
        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if use_cuda else "cpu")

        self._train_data = d_train
        self._test_data = d_test
        self._valid_data = d_val
        self.start_point_seed = start_point_seed
        self._nepochs = params['epochs']
        self._batchsize = params['batch_size']
        self._optimizer = None
        self._loss = None
        self.logs = []
        self.verbose = False

    def _init_model(self):

        torch.manual_seed(self.start_point_seed)
        net_par = {'i_dim': self._train_data.n_var,
                   'o_dim': 1,
                   'h_dim': 10,
                   'n_layers': 1}

        self._model =  network.Net(net_par)
        self._optimizers = torch.optim.Adam(self._model.parameters(), lr=0.001)
        self._LR_multiplier_list = []
        self.logs = []

    def train(self):
        for epoch in range(self._nepochs):
            for (x, y) in self._train_data:
                y_pred = self.predict(x)
                loss = self.compute_loss(y_pred, y, x)
                self.propagate_loss(loss)
            self.print_report(epoch)
            self.validation_step(epoch)

    def opt_lr_rate(self):
        '''
        Optimize the Lagrangian step size. Should we run only 1 time for data of same violation ratio, 
        and  same number of training samples,

        :return: set class object with optimal logs, and optimal model with optimal lr 
        '''
        model_list = []
        lr_list = [1e-3, 5 * 1e-3, 1e-2, 5 * 1e-2, 1e-1]
        val_mae_list = []
        val_vc_list = []
        model_list = []
        logs_list = []
        _LR_multiplier_list_list = []
        for lr in lr_list:
            self._init_model()
            self._LR_rate = lr
            self.train(options = {'mult_fixed':False})
            val_mae_list.append(copy.deepcopy(self.logs[-1][0]))  # use deep copy for safety reason
            val_vc_list.append(copy.deepcopy(self.logs[-1][2]))
            model_list.append(copy.deepcopy(self._model))
            logs_list.append(copy.deepcopy(self.logs))
            _LR_multiplier_list_list.append(copy.deepcopy(self._LR_multiplier_list))

        self.model_list = copy.deepcopy(model_list)
        self.val_mae_list = copy.deepcopy(val_mae_list)
        self.val_vc_list = copy.deepcopy(val_vc_list)

        val_mae_list = [(x - min(val_mae_list)) / float(max(val_mae_list) - min(val_mae_list)) for x in val_mae_list]
        if max(val_vc_list) > min(val_vc_list):
                val_vc_list = [(x - min(val_vc_list)) / float(max(val_vc_list) - min(val_vc_list)) for x in val_vc_list]

        metric_list = np.asarray(val_mae_list) + np.asarray(val_vc_list)

        best_index = [idx for idx in range(5) if metric_list[idx] == np.min(metric_list)][0]

        self._model = model_list[best_index]
        self.logs = logs_list[best_index]
        self._LR_rate = lr_list[best_index]
        self.logs_list = logs_list
        self._LR_multiplier_list_list = _LR_multiplier_list_list
        self._LR_multiplier_list = _LR_multiplier_list_list[best_index]


    def predict(self, x):
        return self._model(x)

    def propagate_loss(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def validation_step(self, epoch):
        #mae_res = list()
        violated_const_model = 0
        violated_const_dataset = 0
        (X, y) = self._valid_data._dataset
        y_pred = self._model.predict(Ten(X))
        val_mae = util.mae(y, y_pred)
        violated_const_dataset += len(util.violated_const(X, y))
        violated_pairs = util.violated_const(X, y_pred)
        violated_const_model += len(violated_pairs)

        y_pred = copy.deepcopy( np.array([10 ** -t for t in y_pred]))
        y_true = copy.deepcopy( np.array([10 ** -t for t in y]))

        val_rmse = np.sqrt( mse(y_true, y_pred))
        y_diff = np.abs(y_true - y_pred)
        sum_mag_viol = np.sum([abs( y_pred[i] -y_pred[j]) for (i,j) in violated_pairs]  )
        median_ae = np.median(y_diff)
        mean_ae = np.mean(y_diff)

        if self.verbose:
            print(f"epoch: {epoch}, "
                  f"MAE: {val_mae}, "
                  f"Violated constraints dataset: {violated_const_dataset}, "
                  f"Violated constraints model: {violated_const_model}")

        self.logs.append([val_mae, violated_const_dataset,  violated_const_model, median_ae, mean_ae, val_rmse, sum_mag_viol])
    def print_report(self, epoch):
        pass

    def test(self):
        #mae_list = list()
        violated_const_dataset = 0
        violated_const_model = 0
        (X, y) = self._test_data._dataset
        y_pred = self._model.predict(Ten(X))

        test_mae = util.mae(y, y_pred)
        violated_const_dataset += len(util.violated_const(X, y))
        violated_pairs = util.violated_const(X, y_pred)
        violated_const_model += len(violated_pairs)

        y_pred = copy.deepcopy(np.array([10 ** -t for t in y_pred]))
        y_true = copy.deepcopy(np.array([10 ** -t for t in y]))

        y_diff = np.abs(y_true - y_pred)
        median_ae = np.median(y_diff)
        mean_ae = np.mean(y_diff)
        rmse_ = np.sqrt(mse(y, y_pred))
        sum_mag_viol = np.sum([abs(y_pred[i] - y_pred[j]) for (i, j) in violated_pairs])

        #print(f"Test precision: {test_mae}, "
        #      f"Violated constraints Dataset: {violated_const_dataset}, "
       #       f"Violated constraints Model: {violated_const_model}, "
        #      f"Duplicates: {duplicates(y_pred)}")
        self.y_true  = y_true
        self.y_pred = y_pred
        self.violated_pairs = violated_pairs
        return (test_mae, violated_const_model, violated_const_dataset, median_ae, mean_ae, rmse_, sum_mag_viol )
        #return (test_mae, violated_const_model, median_ae, mean_ae, rmse_)


    def build_kwb_matrix(self, data):
        """ Build matrix containg dominance informatins

        Every couple (dominant, dominated) is tracked in the matrix.
        Each row represents a couple, while each column a training sample.
        Dominand samples are marked with -1, while the dominated with 1.
        """
        all_couples = util.couples(data.tolist())
        n = max(1, len(all_couples))
        kwb_matrix = np.zeros((n, len(data)))
        for (k, (i, j)) in enumerate(all_couples):
            kwb_matrix[k, i] = -1
            kwb_matrix[k, j] = 1
        return Ten(kwb_matrix.T)

    def compute_loss(self, y_pred, y, x=None):
        pass

    def plot(self):
        pass
