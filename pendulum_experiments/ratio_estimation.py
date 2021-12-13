# input (real data, policy)
import torch
import numpy as np


class DataSet:
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index]



def logireg_loss(de_data, nu_data): # weight decay around 0.1
    # min [- (1/N_de)*sum_{j} log (1/(1+r(x_de^j)))   - (1/N_nu)*sum_{i} log (r(x_nu^i)/(1+r(x_nu^i))) ]
    return -torch.sum(-torch.log(1.+de_data))/de_data.shape[0] -torch.sum(-torch.log(1.+nu_data) + torch.log(nu_data))/nu_data.shape[0]
def ulsif_loss(de_data, nu_data):
    # min [(0.5/N_de)*sum_{j} r(x_de^j)*r(x_de^j) - (1/N_nu)*sum_{i} r(x_nu^i) ]
    return 0.5*torch.sum((de_data)**2)/de_data.shape[0] - torch.sum(nu_data)/nu_data.shape[0]


def rulsif_loss(de_data, nu_data, alpha=0.1):
    # min [(0.5/N_de)*(1-alpha)*sum_{j} r(x_de^j)*r(x_de^j) + (0.5/N_nu)*alpha*sum_{i} r(x_nu^i)*r(x_nu^i) - (1/N_nu)*sum_{i} r(x_nu^i) ]
    return 0.5*(1.-alpha)*torch.sum(de_data**2)/de_data.shape[0] + 0.5*alpha*torch.sum(nu_data**2)/nu_data.shape[0] - torch.sum(nu_data)/nu_data.shape[0]


#def kliep_loss(de_data, nu_data):
#    # min [- (1/N_nu)*sum_{i} log r(x_nu^i) + (1/N_de)*sum_{j}r(x_de^j)]
#    return - torch.sum(torch.log(nu_data))/nu_data.shape[0] + torch.sum(de_data)/de_data.shape[0]


def normalized_logireg_loss(de_data, nu_data):
    # r(x) = c*q(x) is density ratio
    # s.t. 1=(1/N_de)*c*sum_{j}q(x_de^j) <==> s.t. [c=N_de/(sum_{j}q(x_de^j)
    # min [- (1/N_de)*sum_{j} log (1/(1+c*q(x_de^j)))   - (1/N_nu)*sum_{i} log (c*q(x_nu^i)/(1+c*q(x_nu^i))) ] s.t. [1=(1/N_de)*c*sum_{j}q(x_de^j)]
    c = de_data.shape[0]/torch.sum(de_data)
    return -torch.sum(-torch.log(1.+c*de_data))/de_data.shape[0] -torch.sum(-torch.log(1.+c*nu_data) + torch.log(c*nu_data))/nu_data.shape[0]

#def normalized_lsif_loss(de_data, nu_data):
#    # min [(0.5/N_de)*sum_{j} c*c*q(x_de^j)*q(x_de^j) - (1/N_nu)*sum_{i} c*q(x_nu^i) ] s.t. [1=(1/N_de)*c*sum_{j}q(x_de^j)]
#    # min c*[0.5*sum_{j}q(x_de^j)*q(x_de^j)/(sum_{j'}q(x_de^{j'}) - (1/N_nu)*sum_{i} q(x_nu^i) ] s.t. [1=(1/N_de)*c*sum_{j}q(x_de^j)]
#    c = de_data.shape[0]/torch.sum(de_data)
#    return 0.5*torch.sum((c*de_data)**2)/de_data.shape[0] - torch.sum(c*nu_data)/nu_data.shape[0]
#def normalized_kliep_loss(de_data, nu_data):
#    # min [- (1/N_nu)*sum_{i} log c*q(x_nu^i)  + (1/N_de)*sum_{j}r(x_de^j)] s.t. [1=(1/N_de)*c*sum_{j}q(x_de^j)]
#    # <==> min [- (1/N_nu)*sum_{i} log q(x_nu^i) - log c ]
#    # <==> min [- (1/N_nu)*sum_{i} log q(x_nu^i) + log(sum_{j}q(x_de^j)) ]
#    return - torch.sum(torch.log(nu_data))/nu_data.shape[0] + torch.log(torch.sum(de_data))



from model import RatioModel

class RatioEstimation():
    def __init__(self, real_data, H=16):

        self.real_data = torch.from_numpy(real_data.astype(np.float32)).clone()
        self.input_dim = real_data.shape[1]
        self.real_data_num = real_data.shape[0]
        self.H = H

        
        self.model = RatioModel(self.input_dim, self.H)

        #self.loss_fn = rulsif_loss
        #self.loss_fn = normalized_logireg_loss
        self.loss_fn = logireg_loss
        #self.loss_fn = ulsif_loss

        print("[dens] input_dim =",self.input_dim)
        print("[dens] real_data_num =",self.real_data_num)


    def load_sim_data(self, sim_data):
        self.sim_data  = torch.from_numpy(sim_data.astype(np.float32)).clone()
        self.sim_data_num  = sim_data.shape[0]
        print("[dens] sim_data_num =",self.sim_data_num)

    def train_model(self, num_iter=1000, lr=1e-3, weight_decay=.01, holdout_ratio=0.0):
        #self.model = RatioModel(self.input_dim, self.H)
        real_index = np.arange(self.real_data.shape[0])
        np.random.shuffle(real_index)
        sim_index  = np.arange(self.sim_data.shape[0])
        np.random.shuffle(sim_index)

        real_train_data = self.real_data[real_index[int(holdout_ratio*self.real_data.shape[0]):]]
        sim_train_data = self.sim_data[sim_index[int(holdout_ratio*self.real_data.shape[0]):]]

        train_real_dataset = DataSet(real_train_data)
        temp_real_loader = torch.utils.data.DataLoader(train_real_dataset, batch_size=32, shuffle=True, drop_last=True)

        if holdout_ratio>0.001:
            real_test_data  = self.real_data[real_index[:int(holdout_ratio*self.real_data.shape[0])]]
            sim_test_data  = self.sim_data[sim_index[:int(holdout_ratio*self.real_data.shape[0])]]

        train_sim_dataset = DataSet(sim_train_data)
        temp_sim_loader = torch.utils.data.DataLoader(train_sim_dataset, batch_size=32, shuffle=True, drop_last=True)


        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_loss = 1.e12
        update_num = 0
        print("weight_decay",weight_decay)
        for epoch in range(num_iter):

            train_loss =0.
            for nu__data in temp_sim_loader:
                optimizer.zero_grad()
                de__data = iter(temp_real_loader).next()
                loss = self.loss_fn(self.model(de__data), self.model(nu__data))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * de__data.shape[0]

            if holdout_ratio>0.001:
                with torch.no_grad():
                    valid_loss = self.loss_fn(self.model(real_test_data), self.model(sim_test_data)).numpy()
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




    def output_weight(self):
        unnormalized_density_ratio = (torch.nn.functional.relu(self.model(self.real_data))).detach().numpy()
        #return unnormalized_density_ratio
        return unnormalized_density_ratio / unnormalized_density_ratio.mean()

    def save_model(self, filename='temp_ratio_model.pt'):
        torch.save(self.model, filename)

    def load_model(self, filename='temp_ratio_model.pt'):
        self.model = torch.load(filename)



if __name__ == '__main__':

    obac_data   = np.loadtxt('np_obac.csv',delimiter=',')
    sim_data_example = np.loadtxt("np_obac_simulation.csv",delimiter=',' )
    ratio_model = RatioEstimation(obac_data)
    ratio_model.load_sim_data(sim_data_example)

    ratio_model.train_model()

    np.savetxt('weight.csv',ratio_model.output_weight(),delimiter=',')
    import matplotlib.pyplot as plt
    plt.plot(ratio_model.output_weight())
    plt.show()

