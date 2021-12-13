import numpy as np
import torch

import matplotlib.pyplot as plt
"""
obac_data = np.loadtxt('np_obac.csv',delimiter=',')
nll_data = np.loadtxt('temp_unweighted_nll.csv',delimiter=',')


plt.xlim([-4.1*np.pi,4.1*np.pi])
plt.ylim([-15,15])
#plt.scatter(x=obac_data[:,0], y=obac_data[:,1], c=np.log(nll_data - np.min(nll_data) + 1e-2), cmap='jet')
plt.scatter(x=obac_data[:,0], y=obac_data[:,1], c=nll_data, cmap='jet')
plt.colorbar()
plt.show()
"""


class DataSet:
    def __init__(self, X, t):
        self.X = X
        self.t = t

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.t[index]


class NLLRegression():
    def __init__(self, input_data, output_data, hidden_unit_num=16, ob_dim=None):

        # common part
        self.input_data  = torch.from_numpy(input_data.astype(np.float32)).clone()
        self.data_num   = input_data.shape[0]
        self.input_dim  = input_data.shape[1]
        self.output_data = torch.from_numpy(output_data.astype(np.float32)).clone().reshape(self.data_num,1)
        self.output_dim = 1

        #self.output_mean    = torch.sum(self.output_data)/self.data_num
        #self.output_std = torch.sqrt(torch.sum((self.output_data-self.output_mean)**2)/self.data_num)

        self.output_data_min = torch.min(self.output_data)
        self.output_data2 = torch.log(self.output_data - self.output_data_min + 1.e-3)

        #self.output_data2 = (self.output_data - self.output_mean) / self.output_std


        from model import NLLModel
        self.model = NLLModel(self.input_dim, hidden_unit_num)


    #def train_model(self, num_iter=2000, batch_size=32, lr=1e-2, weight_decay=0.001):
    def train_model(self, num_iter=2000, batch_size=32, lr=1e-2, weight_decay=0.001):

        print("[nll] learning rate",lr)
        print("[nll] weight decay",weight_decay)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        train_dataset2 = DataSet(self.input_data, self.output_data2)
        temp_loader = torch.utils.data.DataLoader(train_dataset2, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(num_iter):

            for data in temp_loader:
                X = data[0]
                t = data[1]
                optimizer.zero_grad()
                #loss = torch.sum(torch.abs(self.model(X)-t))
                loss = torch.sum((self.model(X)-t)**2)
                loss.backward()
                optimizer.step()
            if (epoch)%100==0:
                with torch.no_grad():
                    print("epoch ",int((epoch+1)*100/num_iter),"[%]",torch.sum((self.model(self.input_data)-self.output_data2)**2).item())

    def pred(self,obac):
        with torch.no_grad():
            #return ((self.model( torch.from_numpy(obac.astype(np.float32)).clone() ) + self.output_mean) * self.output_std ).numpy()[:,0]
            pred, _ = self.model( torch.from_numpy(obac.astype(np.float32)).clone() )
            pred = torch.exp(pred) + self.output_data_min #- 1.e-3
            return pred.numpy()#[:,0]

"""
temp_model = NLLRegression(obac_data, nll_data)
temp_model.train_model()

nll_pred = np.abs(temp_model.pred(obac_data) -nll_data)

plt.xlim([-4.1*np.pi,4.1*np.pi])
plt.ylim([-15,15])
plt.scatter(x=obac_data[:,0], y=obac_data[:,1], c=np.log(nll_pred+1.e-4), cmap='jet')
plt.colorbar()
plt.show()
"""

