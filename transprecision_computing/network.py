""" strongly inspired by Ferdinando Fioretto's code """

import torch
import torch.nn as nn

Ten = torch.FloatTensor


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        idim = params['i_dim']
        odim = params['o_dim']
        hdim = params['h_dim']
        self._nlayers = params['n_layers']
        self._af = nn.ReLU
        self._of = nn.Linear

        self.i_layer = nn.Sequential(
            nn.Linear(idim, hdim),
            self._af(inplace=True))

        layers = []
        for i in range(self._nlayers-1):
            layers.append(nn.Linear(hdim, hdim))
            layers.append(self._af(inplace=True))
        self.h_layers = nn.Sequential(*layers)

        self.o_layer = nn.Sequential(
            nn.Linear(hdim, odim))

    def forward(self, x):
        o = self.i_layer(x)
        o = self.h_layers(o)
        return self.o_layer(o)

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().numpy()
