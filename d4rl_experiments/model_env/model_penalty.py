import torch
import numpy as np

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ModelPenalty:
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

                                 torch.nn.Linear(hidden_num, 1)#,
                                 #torch.nn.Softplus()
                             )

        self.model = self.model.to(TORCH_DEVICE)
        self.hold_out_ratio=hold_out_ratio
        self.early_stop_num=early_stop_num

    def train(self, train_x, train_y, batch_size=512, reset_flag=True):
        if reset_flag:
             self.__init__(train_x.shape[1], self.hold_out_ratio, self.early_stop_num)

        train_x = torch.from_numpy(train_x.astype(np.float32))
        train_y = torch.from_numpy(train_y.astype(np.float32))
        train_x = train_x[train_y>1.e-12] # remove zero ouput for log transformation
        train_y = train_y[train_y>1.e-12].unsqueeze(-1)

        print("\n[ PenaltyModel ] ")
        print( "pointwise_loss","train_y.shape", train_y.shape, 
               "train_y.min() =", train_y.min(), "train_y.max() =", train_y.max(), 
               "np.median(train_y) =", np.median(train_y), "train_y.mean() =", train_y.mean())

        train_y = torch.log(train_y) # log transformation

        lr = 0.005


        indx = np.random.permutation(train_x.shape[0])
        valid_x = train_x[indx[:int(self.hold_out_ratio*indx.shape[0])]]
        valid_y = train_y[indx[:int(self.hold_out_ratio*indx.shape[0])]]
        train_x = train_x[indx[int(self.hold_out_ratio*indx.shape[0]):]]
        train_y = train_y[indx[int(self.hold_out_ratio*indx.shape[0]):]]


        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(train_x, train_y)
        valid_dataset = TensorDataset(valid_x, valid_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


        total_iter = int(train_x.size(0)/batch_size)
        print(self.model)
        best_valid_loss=1.e12
        update_count = 0
        for i in range(10000):

            for x_batch_t, y_batch_t in train_loader:
                optimizer.zero_grad()
                loss = ( (self.model(x_batch_t.to(TORCH_DEVICE)) - y_batch_t.to(TORCH_DEVICE))**2 ).mean()
                loss.backward()
                optimizer.step()


            with torch.no_grad():
                valid_loss = 0.
                for x_batch_v, y_batch_v in valid_loader:
                    loss = ( (self.model(x_batch_v.to(TORCH_DEVICE)) - y_batch_v.to(TORCH_DEVICE))**2).mean()
                    valid_loss += loss.item() * x_batch_v.shape[0]
                valid_loss /= valid_x.shape[0]


            if valid_loss < best_valid_loss:
                update_count=0
                best_valid_loss=valid_loss
            else:
                update_count+=1

            print(i,valid_loss,update_count,end="\n")
            if update_count>=self.early_stop_num:
                print(i,valid_loss,end="\n")
                break
    

    def predict(self, inputs):
        with torch.no_grad():
            inputs = torch.from_numpy(inputs.astype(np.float32)).to(TORCH_DEVICE)
            return torch.exp(self.model(inputs)).clamp(0., 1.e16).cpu().numpy()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

