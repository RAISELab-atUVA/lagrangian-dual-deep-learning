""" Modified the lagrangian term """

from agent import *
from torch import nn

class SBRregressor2(AbstractAgent):
    def __init__(self, params, d_train, d_test, d_val, start_point_seed):
        super(SBRregressor2, self).__init__(params, d_train, d_test, d_val, start_point_seed)


        self._loss = nn.MSELoss()
        #self.initialize_LR_multipliers()
        self._LR_rate =  None
        self._violations_epoch = []
        self.const_avg_batch = 0
        self._LR_multiplier_list = []


    def initialize_LR_multipliers(self, options):
        self._LR_multipliers = []
        const_batch = []
        for (x, y) in self._train_data:
            n = max(1, len(util.couples(x.tolist())))
            const_batch.append(n)
            if options['mult_fixed']:
                # all Lagrangian multipliers set to be one, and will not be updated during optimization
                self._LR_multipliers.append(np.ones(n))
            else:
                self._LR_multipliers.append(np.zeros(n))
        self.const_avg_batch = int(np.mean(np.array(const_batch)))
        self._LR_multipliers = np.concatenate(self._LR_multipliers)

    # @property
    def const_avg_batch(self):
        return self.const_avg_batch()

    def train(self, options):

        super()._init_model()

        self.initialize_LR_multipliers(options)

        for epoch in range(self._nepochs):
            violations_epoch = []
            offset = 0
            for (x, y) in self._train_data:
                M = self.build_kwb_matrix(x)
                y_pred = self.predict(x)
                loss, violations = self.compute_loss(M, offset, y_pred, y, x)
                self.propagate_loss(loss)
                violations_epoch += violations
                offset += len(violations)
            if not options['mult_fixed']:
                self.update_LR_multipliers(violations_epoch)

            self.validation_step(epoch)
            self.print_report(epoch)

    ''' Override the compute_loss method in AbstractAgent class'''

    def compute_loss(self, M, offset, y_pred, y, x=None):
        loss = self._loss(y_pred, y)
        # g(x') - g(x'') where g(x'') is dominant over g(x'), hence should be have grater value,
        # (N.B. we are considering the -log10 of the error otherwise it would be g(x'') - g(x')
        rules = torch.mm(torch.transpose(M, 0, 1), y_pred)
        # filter only the positive values (violations), therefore the violations to the constraint
        rules = torch.max(Ten(np.zeros(rules.size()[0])), torch.transpose(rules, 0, 1))[0]
        # LR * max(0, g(x') - g(x''))
        lr_mults = Ten([self._LR_multipliers[offset + i] for i in range(len(rules))])
        loss += torch.sum(lr_mults * rules)
        # The below code seems  a bug, ignoring the contribution of violated constraints during optimization
        #
        #loss += torch.sum(Ten([self._LR_multipliers[offset + i] * rules[i] for i in range(len(rules))]))
        violations = [x.item() for x in rules]

        return loss, violations

    ''' Override the propagate_loss method in AbstractAgent class'''

    def propagate_loss(self, loss):
        self._optimizers.zero_grad()
        loss.backward(retain_graph=True)
        self._optimizers.step()

    ''' Override predict method in AbstractAgent class'''

    def predict(self, x):
        # Predict on each data partition
        y_pred = self._model(x)
        return y_pred

    ''' Update the Lagrangian Multipliers associated to the constraint violations'''

    def update_LR_multipliers(self, violations):
        for i in range(len(violations)):
            self._LR_multipliers[i] = self._LR_multipliers[i] + (self._LR_rate * violations[i])

    def print_report(self, epoch):
        self._LR_multiplier_list.append(copy.deepcopy(self._LR_multipliers))
        # a single element contains a vector of lambda_{i,j} at epoch time t
        pass

