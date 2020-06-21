from utils import *

class SharedNet(nn.Module):
  def __init__(self, params):
    super(SharedNet, self).__init__()
    self.input_size = params['input_size']
    self._af = nn.ReLU
    self._of = nn.Sigmoid()


    # 1.Define the shared feature extractor for data of different groups
    self.gen_net = nn.Sequential()
    gen_nnodes_list = [self.input_size] + params['gen_nnodes_list']
    gen_layers = []
    for i in range(len(gen_nnodes_list)-1):
      gen_layers.append(nn.Linear(gen_nnodes_list[i], gen_nnodes_list[i+1]))
      gen_layers.append(self._af(inplace=True))
    self.gen_net = nn.Sequential(*gen_layers)

    z1_nnodes_list = gen_nnodes_list[-1:] + params['z1_nnodes_list']

    # 2. Layer for Men(z=1) classifier
    z1_layers = []
    for i in range(len(z1_nnodes_list)-1):
      z1_layers.append( nn.Linear(z1_nnodes_list[i], z1_nnodes_list[i+1] ))
      z1_layers.append(self._af(inplace = True))
    z1_layers.append( nn.Linear(z1_nnodes_list[-1], 1 ))
    # z1_layers.append(self._of)

    self.z1_net = nn.Sequential(*z1_layers)

    # 3. Layer for Women(z=1) classifier

    z0_nnodes_list = gen_nnodes_list[-1:] + params['z0_nnodes_list']
    z0_layers = []
    for i in range(len(z0_nnodes_list)-1):
      z0_layers.append( nn.Linear(z0_nnodes_list[i], z0_nnodes_list[i+1] ))
      z0_layers.append(self._af(inplace = True))
    z0_layers.append( nn.Linear(z0_nnodes_list[-1], 1 ))
    # z0_layers.append(self._of) We will add this to forward function later
    self.z0_net = nn.Sequential(*z0_layers)

  def forward(self, x, i):
    if len(x) == 0:
      return None
    gen_latent = self.gen_net(x)
    if i == 0:
      score_func = self.z0_net(gen_latent)
    elif i == 1:
      score_func =  self.z1_net(gen_latent)
    else:
        print('Warning, check sensitive input  !')

    return self._of(score_func), score_func

  def predict(self, x, i):

    y_pred,_  = self.forward(x, i)

    # we will return a numpy array here
    return (y_pred.cpu().detach().numpy() >=0.5).astype(int)