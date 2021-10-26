import torch
import numpy as np

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ModelRatio:
    def __init__(self, input_dim, hold_out_ratio, early_stop_num):

        hidden_num=256
        self.model = torch.nn.Sequential(
                                 torch.nn.Linear(input_dim, hidden_num),
                                 torch.nn.Tanh(),
                                 #torch.nn.SiLU(),

                                 torch.nn.Linear(hidden_num, hidden_num),
                                 torch.nn.Tanh(),
                                 #torch.nn.SiLU(),

                                 torch.nn.Linear(hidden_num, hidden_num),
                                 torch.nn.Tanh(),
                                 #torch.nn.SiLU(),

                                 torch.nn.Linear(hidden_num, hidden_num),
                                 torch.nn.Tanh(),
                                 #torch.nn.SiLU(),

                                 torch.nn.Linear(hidden_num, 1),
                                 torch.nn.Softplus()
                             )

        self.hold_out_ratio=hold_out_ratio
        self.early_stop_num=early_stop_num
        self.model = self.model.to(TORCH_DEVICE)

    def train(self, train_x, train_y, batch_size=512, reset_flag=True):
        if reset_flag:
             self.__init__(train_x.shape[1], self.hold_out_ratio, self.early_stop_num)

        train_x = torch.from_numpy(train_x.astype(np.float32))
        train_y = torch.from_numpy(train_y.astype(np.float32))        
        indx = np.random.permutation(train_x.shape[0])
        valid_x = train_x[indx[:int(self.hold_out_ratio*indx.shape[0])]]
        valid_y = train_y[indx[:int(self.hold_out_ratio*indx.shape[0])]]
        train_x = train_x[indx[int(self.hold_out_ratio*indx.shape[0]):]]
        train_y = train_y[indx[int(self.hold_out_ratio*indx.shape[0]):]]


        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)

        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(train_x, train_y)
        valid_dataset = TensorDataset(valid_x, valid_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


        total_iter = int(train_x.size(0)/batch_size)
        print("\n[ RatioModel ] ")
        print(self.model)
        best_valid_loss=1.e12
        update_count = 0
        for i in range(10000):

            for x_batch_t, y_batch_t in train_loader:
                optimizer.zero_grad()
                r1_ = self.model(x_batch_t.to(TORCH_DEVICE))
                r1 = r1_ + 1.e-30*torch.ones_like(r1_)
                r2_ = self.model(y_batch_t.to(TORCH_DEVICE))
                r2 = r2_ + 1.e-30*torch.ones_like(r2_)
                loss1 =  -(torch.log(torch.ones_like(r1)/(r1+torch.ones_like(r1)))).mean()
                loss2 =  -(torch.log(r2/(r2+torch.ones_like(r2)))).mean()
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                valid_loss = 0.
                for x_batch_v, y_batch_v in valid_loader:
                    r1_ = self.model(x_batch_v.to(TORCH_DEVICE))
                    r1 = r1_ + 1.e-30*torch.ones_like(r1_)
                    r2_ = self.model(y_batch_v.to(TORCH_DEVICE))
                    r2 = r2_ + 1.e-30*torch.ones_like(r2_)
                    loss1 =  -(torch.log(torch.ones_like(r1)/(r1+torch.ones_like(r1)))).sum()
                    loss2 =  -(torch.log(r2/(r2+torch.ones_like(r2)))).sum()
                    valid_loss += (loss1 + loss2).item()
                valid_loss /= valid_x.shape[0]

            print(i,valid_loss)
            if valid_loss < best_valid_loss:
                update_count=0
                best_valid_loss=valid_loss
            else:
                update_count+=1
            if update_count>=self.early_stop_num:
                print(i,valid_loss,end="\n")
                break
    

    def predict(self, inputs):
        inputs = torch.from_numpy(inputs.astype(np.float32))
        with torch.no_grad():
            r1_ = self.model(inputs.to(TORCH_DEVICE))
            r1 = r1_ + 1.e-30*torch.ones_like(r1_)
            return r1.cpu().numpy()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    data = np.loadtxt("for_debug/np_obac.csv",delimiter=',') 
    data2 = np.loadtxt("for_debug/np_obac_sim.csv",delimiter=',')
    print(data.shape,data2.shape)
    
    model = ModelRatio(data.shape[1])
    model.train(data,data2)
    w = model.predict(data)

    import matplotlib.pyplot as plt
    plt.plot(data2[:,0], data2[:,1],"kx",markersize=4)
    plt.scatter(x=data[:,0], y=data[:,1], c=np.log10(w), cmap='jet', vmin=-np.max(np.log10(w)), vmax=np.max(np.log10(w)))
    plt.colorbar()
    plt.show()
