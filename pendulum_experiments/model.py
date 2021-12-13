import torch
import torch.nn.functional as F

class DynamicsModel(torch.nn.Module): # transitioin function
    def __init__(self, D_in, D_out, hidden_unit_num):

        print("[DynamicsModel] H =",hidden_unit_num)
        super(DynamicsModel, self).__init__()

        # zero hidden layer
        #self.l1 = torch.nn.Linear(D_in, D_out, bias=False)

        # one hidden layer
        self.l1 = torch.nn.Linear(D_in, hidden_unit_num)
        self.l2 = torch.nn.Linear(hidden_unit_num, D_out) # , bias=False

        self.logvar = torch.nn.Parameter(torch.zeros(D_out), requires_grad=True)


        # two hidden layer
        #self.l1 = torch.nn.Linear(D_in, hidden_unit_num)
        #self.l2 = torch.nn.Linear(hidden_unit_num, hidden_unit_num)
        #self.l3 = torch.nn.Linear(hidden_unit_num, D_out)

    def forward(self, X):

        mu = self.l2(torch.tanh(self.l1(X)))
        return self.l2(torch.tanh(self.l1(X))), self.logvar*torch.ones_like(mu)
        #return self.l2(F.relu(self.l1(X)))

        #return self.l3(torch.tanh(self.l2(torch.tanh(self.l1(X)))))
        #return self.l3(F.relu(self.l2(F.relu(self.l1(X)))))



class RatioModel(torch.nn.Module): # density ratio
    def __init__(self, D_in, hidden_unit_num):
        super().__init__()
        print("[RatioModel] H =",hidden_unit_num)

        #self.l1 = torch.nn.Linear(D_in, hidden_unit_num)
        #self.l2 = torch.nn.Linear(hidden_unit_num, 1) # output dimension is always 1.

        self.l1 = torch.nn.Linear(D_in, hidden_unit_num)
        self.l2 = torch.nn.Linear(hidden_unit_num, hidden_unit_num)
        self.l3 = torch.nn.Linear(hidden_unit_num, 1)

    def forward(self, X):
        #return F.softplus(self.l2(torch.tanh(self.l1(X))))
        return F.softplus(self.l3(torch.tanh(self.l2(torch.tanh(self.l1(X))))))



class GradLSDModel(torch.nn.Module): # gradient of log-stationary distribution
    def __init__(self, D_in, D_out):
        super().__init__()
        self.l1 = torch.nn.Linear(D_in, D_out)

    def forward(self, X):
        return self.l1(X)


class NLLModel(torch.nn.Module): # nll
    def __init__(self, D_in, hidden_unit_num):
        super().__init__()

        print("[NLLModel] H =", hidden_unit_num)
        self.l1 = torch.nn.Linear(D_in, hidden_unit_num)
        #self.l2 = torch.nn.Linear(hidden_unit_num, 1) # , bias=False

        self.l2 = torch.nn.Linear(hidden_unit_num, hidden_unit_num) 
        self.l3 = torch.nn.Linear(hidden_unit_num, 1) 


    def forward(self, X):

        #return self.l2(torch.tanh(self.l1(X)))
        return self.l3(torch.tanh(self.l2(torch.tanh(self.l1(X)))))


