""" strongly inspired by Ferdinando Fioretto's code """
import network, dataset, agent

from torch_inputs import *
from agent import *
class Regressor(AbstractAgent):
    def __init__(self, params, d_train, d_test, d_val, start_point_seed):
        super(Regressor, self).__init__(params,d_train, d_test, d_val,  start_point_seed)
        self._loss = nn.MSELoss()

    def train(self):
        super()._init_model()

        for epoch in range(self._nepochs):
            for (x, y) in self._train_data:
                y_pred = self.predict(x)
                loss = self.compute_loss(y_pred, y, x)
                self.propagate_loss(loss)
            self.validation_step(epoch)

    ''' Override the compute_loss method in AbstractAgent class'''

    def compute_loss(self, y_pred, y, x=None):
        loss = self._loss(y_pred, y)
        return loss

    ''' Override the propoagate_loss method in AbstractAgent class'''

    def propagate_loss(self, loss):
        self._optimizers.zero_grad()
        loss.backward(retain_graph=True)
        self._optimizers.step()

    ''' Override predict method in AbstractAgent class'''

    def predict(self, x):
        # Predict on each data partition
        y_pred = self._model(x)
        return y_pred


