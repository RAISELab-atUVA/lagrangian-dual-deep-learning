from utils import *
from network import *

class LDSharedModel(object):

 def __init__(self, params):
    self.device = params.get('device', 'cuda')
    for key, val in params.items():
        setattr(self, key, val)
    self.input_size = self.X_val.shape[1]
    train_tensor = TensorDataset(Tensor(np.c_[self.X_train, self.Z_train]), Tensor(self.y_train))
    self.train_loader = DataLoader(dataset = train_tensor, batch_size = self.bs, shuffle = True)
    self.fitted_model =  None

 def predict(self, X_test,Z_test):
      """
      Return Prediction for Future Test Data
      """
      if 'numpy' not in str(type(X_test)):
        X_test= X_test.values
      X_test = torch.FloatTensor( X_test[:, :self.input_size]).to(self.device)
      all_pred = []
      for i in range(len(X_test)):
        # use associated branch for making prediction z=0 or z=1 since two branches here
        all_pred.append( self.fitted_model.forward(X_test[i,:], Z_test[i])[0].cpu().data.numpy())
      pred = np.concatenate(all_pred)

      return  (pred>=0.5).astype(int), pred

 def _model_eval(self, model):
     """
     Evaluate current model based on Accuracy and Fairness Score (DI-score)
     """
     self.fitted_model = copy.deepcopy(model)
     y_pred_val, _ = self.predict(self.X_val, self.Z_val)
     acc, p_value, di_score = compute_fairness_score(self.y_val, y_pred_val, self.Z_val)

     return  acc, p_value, di_score

 def fit(self, options):

     #train_verbose = options.get('train_verbose', False)
     seed = options.get('seed', 0)
     model_lr = options.get('model_lr', 1e-2)
     epochs = options.get('epochs', 200)
     step_size = options['step_size']

     lr_mult = options['lr_mult'] # Initial Lagrangian multiplier, 0 for LD method
     return_output = options.get('return_output', False)
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

         acc, p_value, di_score = self._model_eval(model)
         logs.append(copy.deepcopy([acc, di_score, p_value, lr_mult]))

         lr_mult += step_size * np.mean(violation_list)

     self.fitted_model = model
     self.best_options = options
     self.logs = logs
     self.best_acc = acc

     if return_output:
        return model, acc, logs

 def hyper_opt(self, grid_search_list):

     best_metric = -np.inf
     best_model = None
     best_options = None
     best_logs = None
     for options in grid_search_list:
         print( options)
         options['return_output']  = True
         curr_model, curr_acc, logs = self.fit(options)
         if options['acc_only'] :
             curr_metric = curr_acc
         else:
             curr_metric = logs[-1][0] - logs[-1][1] # Acc - DI-score as a heuristic rule to choose the model

         if curr_metric > best_metric:
             best_metric = curr_metric
             best_model = curr_model
             best_options = options
             best_logs = logs

     self.model = best_model
     self.best_metric = best_metric
     self.best_options = best_options
     self.logs = best_logs

