import numpy as np
import torch
from model import DynamicsModel


# transition model: s'-s = f(s,a) + noise

class DataSet:
    def __init__(self, X, t, w):
        self.X = X
        self.t = t
        self.w = w

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.t[index], self.w[index]


class DynamicsRegression():
    def __init__(self, input_data, output_data, hidden_unit_num, ob_dim=None, B_dash=1.):

        self.B_dash=B_dash
        self.hidden_unit_num=hidden_unit_num

        self.input_data  = torch.from_numpy(input_data.astype(np.float32)).clone()
        self.output_data = torch.from_numpy(output_data.astype(np.float32)).clone()
        self.data_num   = input_data.shape[0]
        self.input_dim  = input_data.shape[1]
        self.output_dim = output_data.shape[1]

        if ob_dim is None:
            self.ob_dim = output_data.shape[1]

        self.model = DynamicsModel(self.input_dim, self.output_dim, self.hidden_unit_num)

        self.data_weight = torch.ones(self.data_num,1)
        #self.logvar = torch.nn.Parameter(torch.sum(self.output_data**2, dim=0)/self.input_data.shape[0])

    def save_model(self, filename='temp_mle_dynamics_model'):
        torch.save(self.model, filename+'_param.pt')
        np.savetxt(filename+'_iw.csv', self.data_weight.numpy(),delimiter=',')
        

    def load_model(self, filename='temp_mle_dynamics_model'):
        self.model = torch.load(filename+'_param.pt')
        self.data_weight = torch.from_numpy(np.loadtxt(filename+'_iw.csv',delimiter=',').astype(np.float32)).clone().reshape(self.data_num,1)
        print("load ",filename)

    def get_data_weight(self):
        return self.data_weight.numpy()

    def load_data_weight(self, data_weight):
        print("[reg] load weight")
        self.data_weight = torch.tensor(data_weight.astype(np.float32).reshape(data_weight.shape[0],1))


    def train_model(self, num_iter=10000,#2000, 
                    batch_size=32, 
                    lr=1e-3, weight_decay=0.0001, 
                    param_update_penalty=0.0,
                    holdout_ratio=0.0, 
                    grad_vector=None):
        #self.model = DynamicsModel(self.input_dim, self.output_dim, self.hidden_unit_num)
        init_param=[]
        for p in self.model.parameters():
            init_param.append(torch.flatten(p.data.clone()))
            print("p.data.shape",p)
        init_param = torch.cat(init_param,dim=0)
        print("init_param.shape",init_param.shape)

        print("[reg] learning rate",lr)
        print("[reg] weight decay",weight_decay)
        print("[reg] penalty",param_update_penalty)        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)


        data_index = np.arange(self.data_num)
        np.random.shuffle(data_index)

        train_input_data = self.input_data[data_index[int(holdout_ratio*self.data_num):]]
        train_target_data = self.output_data[data_index[int(holdout_ratio*self.data_num):]]
        train_data_weight = self.data_weight[data_index[int(holdout_ratio*self.data_num):]]

        if holdout_ratio>0.001:
            valid_input_data = self.input_data[data_index[:int(holdout_ratio*self.data_num)]]
            valid_data_weight = self.data_weight[data_index[:int(holdout_ratio*self.data_num)]]
            valid_target_data = self.output_data[data_index[:int(holdout_ratio*self.data_num)]]

        train_dataset2 = DataSet(train_input_data, train_target_data, train_data_weight)
        temp_loader = torch.utils.data.DataLoader(train_dataset2, batch_size=batch_size, shuffle=True, drop_last=True)

        self.grad_vector=grad_vector
        print("grad_vector",grad_vector)

        best_loss = 1.e12
        update_num = 0
        for epoch in range(num_iter):

            train_loss = 0.
            for data in temp_loader:
                X = data[0]
                t = data[1]
                w = data[2]
                optimizer.zero_grad()
                mu, logvar = self.model(X)
                inv_var = torch.exp(-logvar)
                loss = 0.5 * (  (inv_var*((mu-t)**2) + logvar) * w  ).sum() / X.shape[0] # averaged per data

                temp_param=[]
                for p in self.model.parameters():
                    temp_param.append(torch.flatten(p.data))

                temp_param=torch.cat(temp_param,dim=0)
                if self.grad_vector is not None:
                    loss += torch.sum((temp_param-init_param)*self.grad_vector) # averaged per data
                #loss += param_update_penalty*torch.sum((self.grad_vector*(init_param-temp_param))**2)
                loss += param_update_penalty*( (init_param-temp_param)**2 ).sum()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X.shape[0]

            if holdout_ratio>0.001:
                with torch.no_grad():
                    mu, logvar = self.model(valid_input_data)
                    inv_var = torch.exp(-logvar)
                    valid_loss = 0.5*(  (inv_var*((mu-valid_target_data)**2) + logvar) * valid_data_weight  ).sum() / valid_input_data.shape[0]
                    temp_param=[]
                    for p in self.model.parameters():
                        temp_param.append(torch.flatten(p.data))
                    temp_param=torch.cat(temp_param,dim=0)
                    if self.grad_vector is not None:
                        valid_loss += torch.sum((temp_param-init_param)*self.grad_vector) # averaged per data
                    valid_loss += param_update_penalty*( (temp_param-init_param)**2 ).sum()
                if best_loss>valid_loss:
                    update_num = 0
                    best_loss = valid_loss
                else:
                    update_num += 1

                print("epoch, valid_loss, update_num", epoch, valid_loss, update_num)
            else:
                if best_loss>train_loss:
                    update_num = 0
                    best_loss = train_loss
                else:
                    update_num += 1
                print("epoch, train_loss, update_num", epoch, train_loss, update_num)

            if update_num>20:
                break



        temp_param=[]
        for p in self.model.parameters():
            temp_param.append(torch.flatten(p.data))
            print(p.data)
        temp_param=torch.cat(temp_param,dim=0)
        print("parameter_diff",torch.sum((init_param-temp_param)**2))

        with torch.no_grad():
            print("var",torch.exp(self.model.logvar))

        with torch.no_grad():
            mu, logvar = self.model(self.input_data)
            inv_var = torch.exp(-logvar)            
            unweighted_nll = 0.5 * (inv_var*((mu-self.output_data)**2) + logvar ).sum(-1)
            unweighted_nll += 0.5 * np.log(2.*np.pi) * self.ob_dim
        np.savetxt('temp_unweighted_nll.csv', unweighted_nll.numpy()-np.min(unweighted_nll.numpy()), delimiter=',')


    def loss(self):
        X = self.input_data
        t = self.output_data
        w = self.data_weight
        with torch.no_grad():
            mu, logvar = self.model(X)
            inv_var = torch.exp(-logvar)  
            loss = 0.5 * (  (inv_var*((mu-t)**2) + logvar) * w ).sum() / X.shape[0] 
            loss += 0.5 * np.log(2.*np.pi) * self.ob_dim
        return loss.numpy()


    def sim_next_ob(self, ob, ac):

        obac = np.concatenate((ob,ac),axis=0)
        with torch.no_grad():
            pred, logvar = self.model( torch.from_numpy(obac.astype(np.float32)).clone() )
            noise = torch.randn(self.output_dim) * torch.exp(0.5*logvar)
            y = pred + noise
        return ob + y.numpy()

    def get_b_hat(self):
        with torch.no_grad():
            self.noise_weight = torch.exp(self.model.state_dict()["logvar"])
            mu, _ = self.model(self.input_data)
            unweighted_nll = 0.5* (torch.unsqueeze(torch.sum((1./self.noise_weight)*((mu-self.output_data)**2),dim=1),1)+torch.sum(torch.log(2.*np.pi*self.noise_weight)))
        return self.B_dash*0.5/np.sqrt(self.loss() - np.min(unweighted_nll.numpy()))

    #def loss2(self, reward_fn, gamma):
    #    with torch.no_grad():
    #        self.noise_weight = torch.exp(self.model.state_dict()["logvar"])
    #        mu, _ = self.model(self.input_data)
    #        unweighted_nll = 0.5* (torch.unsqueeze(torch.sum((1./self.noise_weight)*((mu-self.output_data)**2),dim=1),1)+torch.sum(torch.log(2.*np.pi*self.noise_weight)))
    #    temp = self.B_dash*torch.sqrt( self.loss() - np.min(unweighted_nll.numpy()) )
    #    for i in range(self.data_num):
    #        temp -= (self.data_weight[i,0]/self.data_num) * reward_fn(self.input_data[i]) / (1.-gamma)
    #    return temp[0]
        

if __name__ == '__main__':
    obac_data   = np.loadtxt('np_obac.csv',delimiter=',')
    nextob_data = np.loadtxt('np_diff_ob.csv',delimiter=',')

    test_model = DynamicsRegression(obac_data, nextob_data, 8)

    #weight = np.loadtxt('weight.csv',delimiter=',')
    #test_model.load_data_weight(weight)
    #test_model.load_model()

    test_model.train_model()

    test_model.save_model()

    
