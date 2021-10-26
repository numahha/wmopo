import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time
import math

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def truncated_standardized_normal(shape, a=-2., b=2.):
    a = torch.Tensor([a])
    b = torch.Tensor([b])

    U = torch.distributions.uniform.Uniform(0, 1)
    u = U.sample(shape)

    Fa = 0.5 * (1 + torch.erf(a/math.sqrt(2)))
    Fb = 0.5 * (1 + torch.erf(b/math.sqrt(2)))
    return math.sqrt(2)*torch.erfinv(2 *((Fb - Fa) * u + Fa) - 1)


def get_affine_params(ensemble_size, in_features, out_features):
    w = truncated_standardized_normal(shape=(ensemble_size, in_features, out_features) ) / (2.0 * math.sqrt(in_features))
    w = torch.nn.Parameter(w)
    b = torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))
    return w, b


class EnsembleModel(torch.nn.Module):

    def __init__(self, ensemble_size, input_num, output_num, hidden_num=200):
        super().__init__()

        self.num_nets = ensemble_size
        self.input_num  = input_num
        self.output_num = output_num

        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size, input_num,  hidden_num)
        self.lin1_w, self.lin1_b = get_affine_params(ensemble_size, hidden_num, hidden_num)
        self.lin2_w, self.lin2_b = get_affine_params(ensemble_size, hidden_num, hidden_num)
        self.lin3_w, self.lin3_b = get_affine_params(ensemble_size, hidden_num, hidden_num)
        self.lin4_w, self.lin4_b = get_affine_params(ensemble_size, hidden_num, hidden_num)
        self.lin5_w, self.lin5_b = get_affine_params(ensemble_size, hidden_num, hidden_num)
        self.lin6_w, self.lin6_b = get_affine_params(ensemble_size, hidden_num, 2*output_num)

        self.inputs_mu = torch.nn.Parameter(torch.zeros(input_num), requires_grad=False)
        self.inputs_sigma = torch.nn.Parameter(torch.zeros(input_num), requires_grad=False)

        self.max_logvar = torch.nn.Parameter(  torch.ones(1, output_num, dtype=torch.float32) / 2.0)
        self.min_logvar = torch.nn.Parameter(- torch.ones(1, output_num, dtype=torch.float32) * 10.0)


    def compute_decays(self):
        loss = 0.
        loss += 1. * (self.lin0_w ** 2).sum() 
        loss += 1. * (self.lin1_w ** 2).sum()
        loss += 1. * (self.lin2_w ** 2).sum()
        loss += 1. * (self.lin3_w ** 2).sum()
        loss += 1. * (self.lin4_w ** 2).sum()
        loss += 1. * (self.lin5_w ** 2).sum()
        loss += 1. * (self.lin6_w ** 2).sum()
        return 0.00001 * loss/ 2.0


    def fit_input_stats(self, data):
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0
        self.inputs_mu.data = torch.from_numpy(mu).to(TORCH_DEVICE).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(TORCH_DEVICE).float()


    def forward(self, inputs):
        inputs = (inputs - self.inputs_mu) / self.inputs_sigma
        inputs = inputs.matmul(self.lin0_w) + self.lin0_b

        inputs = F.silu(inputs)
        inputs = inputs.matmul(self.lin1_w) + self.lin1_b

        inputs = F.silu(inputs)
        inputs = inputs.matmul(self.lin2_w) + self.lin2_b

        inputs = F.silu(inputs)
        inputs = inputs.matmul(self.lin3_w) + self.lin3_b

        inputs = F.silu(inputs)
        inputs = inputs.matmul(self.lin4_w) + self.lin4_b

        inputs = F.silu(inputs)
        inputs = inputs.matmul(self.lin5_w) + self.lin5_b

        inputs = F.silu(inputs)
        inputs = inputs.matmul(self.lin6_w) + self.lin6_b

        mean   = inputs[:, :, :self.output_num]
        logvar = inputs[:, :, self.output_num:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar


class ModelDynaNNEnsemble:
    def __init__(self, inputs, targets, seed, hidden_num, batch_size, hold_out_ratio, early_stop_num):
        self.seed = seed
        self.hidden_num = hidden_num
        self.batch_size = batch_size

        self.hold_out_ratio=hold_out_ratio
        self.early_stop_num = early_stop_num

        self.num_nets   = 7
        self.num_elites = 5
        self.lr = 0.001

        self.model = EnsembleModel(self.num_nets, input_num=inputs.shape[1], output_num=targets.shape[1], hidden_num=self.hidden_num)

        self.model=self.model.to(TORCH_DEVICE)
        self.model.fit_input_stats(inputs)
        self._model_inds = range(self.num_elites)


    def train(self, train_x, train_y, train_w, reset_flag=False):

        if reset_flag is True:
            self.__init__(train_x, train_y, self.seed, self.hidden_num, self.batch_size, self.hold_out_ratio, self.early_stop_num)
        train_x = torch.from_numpy(train_x.astype(np.float32))
        train_y = torch.from_numpy(train_y.astype(np.float32))
        train_w = torch.flatten(torch.from_numpy(train_w.astype(np.float32))).unsqueeze(1)
        indx = np.random.permutation(train_x.shape[0])
        valid_x = train_x[indx[:int(self.hold_out_ratio*indx.shape[0])]]
        valid_y = train_y[indx[:int(self.hold_out_ratio*indx.shape[0])]]
        valid_w = train_w[indx[:int(self.hold_out_ratio*indx.shape[0])]]
        train_x = train_x[indx[int(self.hold_out_ratio*indx.shape[0]):]]
        train_y = train_y[indx[int(self.hold_out_ratio*indx.shape[0]):]]
        train_w = train_w[indx[int(self.hold_out_ratio*indx.shape[0]):]]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        train_dataset = TensorDataset(train_x, train_y, train_w)
        valid_dataset = TensorDataset(valid_x, valid_y, valid_w)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,  drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        print("[ Ensemble NNs ] ")
        print("train_w",train_w.shape,"valid_w",valid_w.shape)
        print("min_train_w",train_w.min(), "max_train_w",train_w.max())
        print("min_valid_w",train_w.min(), "max_valid_w",valid_w.max())
        print("batch_size",self.batch_size)
        print("Num of train[w>1.]",train_w[train_w>1.].shape, "Num of valid[w>1.]",valid_w[valid_w>1.].shape)
        print("hold_out_ratio",self.hold_out_ratio)
        print("lr",self.lr)
        print("hidden_num",self.hidden_num)
        best_valid_losses = np.array([1.e12 for ii in range(self.num_nets)])
        update_count = 0
        start_time = time.time()
        for i in range(10000):

            for x_batch_t, y_batch_t , w_batch_t in train_loader:
                optimizer.zero_grad()
                loss = self.model.compute_decays()
                loss += 0.001 * torch.abs(self.model.max_logvar - self.model.min_logvar).sum()
                mean, logvar = self.model(x_batch_t.to(TORCH_DEVICE))
                train_losses = (((mean - y_batch_t.to(TORCH_DEVICE)) ** 2) * torch.exp(-logvar) + logvar)*w_batch_t.to(TORCH_DEVICE)
                train_losses = train_losses.mean() # [ model_index, data_index, output_dimension ]
                loss += train_losses
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                valid_losses=0.
                for x_batch_v, y_batch_v, w_batch_v in valid_loader:
                    mean, logvar = self.model(x_batch_v.to(TORCH_DEVICE))
                    temp_valid_losses = (((mean - y_batch_v.to(TORCH_DEVICE)) ** 2) * torch.exp(-logvar) + logvar)*w_batch_v.to(TORCH_DEVICE)
                    valid_losses += temp_valid_losses.mean(-1).mean(-1)*x_batch_v.shape[0]

            valid_losses = (valid_losses/valid_x.shape[0]).cpu().numpy()

            update_flag=False
            for ii in range(self.num_nets):
                if valid_losses[ii] < best_valid_losses[ii]:
                    best_valid_losses[ii] = valid_losses[ii]
                    update_flag=True
                    update_count=0

            if (update_flag==False) and (i>self.early_stop_num):
               update_count+=1
            print("Iter", i,"valid_loss",valid_losses.round(4),"sec",int(time.time()-start_time),"update_count", update_count,end="    \n")


            if update_count>=self.early_stop_num:
                print("Iter", i,"temp_valid_loss",valid_losses.round(4),"sec",int(time.time()-start_time))
                print("Iter", i,"best_valid_loss",best_valid_losses.round(4),"sec",int(time.time()-start_time))
                break
        sorted_inds = np.argsort(valid_losses)
        self._model_inds = sorted_inds[:self.num_elites].tolist()
        print("elite_inds =",self._model_inds)
        print("self.min_logvar", self.model.min_logvar)
        print("self.max_logvar", self.model.max_logvar)

    def predict(self, obac):
        with torch.no_grad():
            tobac = torch.from_numpy(obac.astype(np.float32)).to(TORCH_DEVICE)
            mean, logvar = self.model(tobac)
        mean = mean.cpu().numpy()
        std  = np.exp(0.5*logvar.cpu().numpy()) # sigma = exp(0.5*log(sigma^2))
        sample = mean + np.random.normal(size=mean.shape) * std

        num_models, batch_size, _ = mean.shape
        model_inds = np.random.choice(self._model_inds, size=batch_size)
        
        batch_inds = np.arange(0, batch_size)
        sample = sample[model_inds, batch_inds]
        return sample[:,:1], (sample[:,1:]+obac[:,:(sample.shape[1]-1)])

    def get_pointwise_loss(self, obac, rewob2):
        with torch.no_grad():
            tobac   = torch.from_numpy(obac.astype(np.float32)).to(TORCH_DEVICE)
            trewob2 = torch.from_numpy(rewob2.astype(np.float32)).to(TORCH_DEVICE)
            mean, logvar = self.model(tobac)
            losses = ((mean - trewob2) ** 2) * torch.exp(-logvar) + logvar # [ model_index, data_index, output_dimension ]
        return 0.5 * (losses[self._model_inds].sum(-1).mean(0)).cpu().numpy()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("self.min_logvar", self.model.min_logvar)
        print("self.max_logvar", self.model.max_logvar)

    def load_model(self, path):
        print("[before] self.max_logvar", self.model.max_logvar)
        self.model.load_state_dict(torch.load(path))
        print("[after] self.min_logvar", self.model.min_logvar)
        print("[after] self.max_logvar", self.model.max_logvar)


if __name__ == '__main__':
    import gym
    import d4rl
    import numpy as np

    env = gym.make("hopper-medium-replay-v1")
    dataset = env.get_dataset()
    obs = dataset['observations']
    act = dataset['actions']
    next_obs = dataset['next_observations']
    rew = np.expand_dims(dataset['rewards'], 1)
    delta_obs = next_obs - obs
    train_x  = np.concatenate((obs, act), axis=-1)[:1000]
    train_y = np.concatenate((rew, delta_obs), axis=-1)[:1000]
    train_w = np.ones((train_x.shape[0],1))


    mdl = ModelDynaNNEnsemble(train_x, train_y, 
                              0, 32, 32, # seed, hidden, batch
                              0.1, 5) #hold_out_ratio, early_stop_num

    mdl.load_model("test_nn")
    #mdl.train(train_x, train_y, train_w)
    mdl.save_model("test_nn")
    mdl.get_pointwise_loss(train_x[:101], train_y[:101])


