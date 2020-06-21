from utils import *


class LDSharedModel(object):

 def __init__(self, params):
    # we no need to specify X_val, y_val since we provided in params of parent class
    self.device = params.get('device', 'cuda')
    self.X_val = params['X_val']
    self.input_size = self.X_val.shape[1]
    self.Z_val = params['Z_val']
    self.y_val = params['y_val']
    X_train  = params['X_train']
    y_train = params['y_train']
    Z_train = params['Z_train']

    self.bs = params.get('batch_size', 200)  # for safety reason in case I forgot
    train_tensor = TensorDataset(Tensor(np.c_[X_train, Z_train]), Tensor(y_train))
    self.train_loader = DataLoader(dataset=train_tensor, batch_size=self.bs, shuffle=True)
    self.model =  None

 def predict(self, X_test,Z_test):
      # X_test should be a numpy array
      if 'numpy' not in str(type(X_test)):
        X_test= X_test.values
      X_test = torch.FloatTensor( X_test[:, :self.input_size]).to(self.device)
      all_pred = []
      for i in range(len(X_test)):
        # use associated branch for making prediction z=0 or z=1 since two branches here
        all_pred.append( self.fitted_model.forward(X_test[i,:], Z_test[i])[0].cpu().data.numpy())
      pred =   np.concatenate(all_pred)

      return  (pred>=0.5).astype(int), pred

 def _model_eval(self, model):

     self.fitted_model = copy.deepcopy(model)
     self.fitted_model.eval()
     Z_val = self.Z_val
     y_pred, y_soft_pred = self.predict(self.X_val, Z_val)

     y_pred_1 = y_pred[Z_val == 1]
     y_pred_0 = y_pred[Z_val == 0]
     n1, n0 = float(len(y_pred_1)), float(len(y_pred_0))
     p1, p0 = np.sum(y_pred_1) / n1, np.sum(y_pred_0) / n0
     acc = accuracy_score(self.y_val, y_pred)
     p_value = min(p1 / p0, p0 / p1)
     delta_f = abs(np.mean( y_soft_pred[Z_val==0]) - np.mean(y_soft_pred[Z_val==1]))
     # model for evaluation is acc - 3 *DI_score
     return  acc, abs(p1 - p0), p_value, delta_f

 def fit(self, options):

     #train_verbose = options.get('train_verbose', False)
     seed = options.get('seed', 0)
     model_lr = options.get('model_lr', 1e-2)
     epochs = options.get('epochs', 200)
     step_size = options['step_size']

     lr_mult = options['lr_mult'] # Initial Lagrangian multiplier, 0 for LD method
     torch.manual_seed(seed)
     model = SharedNet(options['model_params']).to(self.device)
     bce_criterion = nn.BCELoss(reduce='mean')
     optimizer = torch.optim.Adam(model.parameters(), lr = model_lr)
     logs = []

     for epoch in range(epochs):
         violation_list = []
         for input_train, target_train in self.train_loader:

             model.train()
             input_train = input_train.to(self.device)
             target_train = target_train.to(self.device)
             z_train = input_train[:, self.input_size]
             n = len(z_train)
             loss = torch.tensor(0.0)
             mean_output_list = []
             for i in range(2):
                 optimizer.zero_grad()
                 x_train = input_train[z_train == i, :self.input_size]
                 y_train = target_train[z_train == i]
                 if len(y_train > 1):
                     ni = float(len(y_train))
                     output, score_func = model.forward(x_train, i)
                     loss += ni / n * bce_criterion(output, y_train)
                     mean_output_list.append(torch.mean(output))
             if len(mean_output_list) > 1:
                 # add the mean difference between two groups
                 violation = torch.abs(mean_output_list[0] - mean_output_list[1])
                 loss += lr_mult * violation
                 violation_list.append(violation.item())

             loss.backward()
             optimizer.step()

         acc, di_score, p_value, delta_f = self._model_eval(model)
         logs.append([acc, di_score, p_value, lr_mult, delta_f])

         lr_mult += step_size * np.mean(violation_list) # probably we can replace mean by median

     self.model = model
     self.best_options = options
     self.logs = logs
     self.best_acc = acc
     return model, acc, logs

 def hyper_opt(self, grid_search_list):

     best_acc = -np.inf
     best_model = None
     best_options = None
     best_logs = None
     for options in grid_search_list:
         print( options)
         curr_model, curr_acc, logs = self.fit(options)
         if curr_acc > best_acc:
             best_acc = curr_acc
             best_model = curr_model
             best_options = options
             best_logs = logs

     self.model = best_model
     self.best_acc = best_acc
     self.best_options = best_options
     self.logs = best_logs

