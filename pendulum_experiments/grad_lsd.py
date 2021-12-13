import numpy as np
import torch


def feature(ob):
    return torch.cat([torch.ones(1),ob,ob**2])


class GradLSD():

    def __init__(self, dynamics_model,
                 policy, init_dist, termination, gamma, 
                 penalty=0.1, lam=0.9, num_iter=2000): 

        with torch.no_grad():
            self.noise_weight=torch.exp(dynamics_model.model.logvar)
        self.model=dynamics_model.model
        self.gamma=gamma

        # LSTD (lambda)
        print("[lstd] lambda =",lam)
        print("[lstd] penalty =",penalty)

        dummy_optimizer = torch.optim.SGD(self.model.parameters(), lr=1.)
        dummy_optimizer.zero_grad()

        def get_sim_one_episode():
            ob = init_dist()
            a_mat=0
            b_vec=0
            e_vec=0

            c_mat=0
            phi_sum=0
            t=0
            while 1:
                # transition
                ac = policy(ob)
                ob_torch = torch.from_numpy(ob.astype(np.float32))
                pred, logvar = self.model( torch.from_numpy(np.concatenate((ob,ac),axis=0).astype(np.float32)) )
                noise = torch.randn(ob.shape[0]) * torch.exp(0.5*logvar)
                next_ob = ob_torch + pred.clone().detach() + noise.detach()

                # compute gradient as cost
                loss = - 0.5*torch.sum(   torch.exp(-logvar) * ((ob_torch+pred - next_ob)**2) + logvar )
                loss.backward()
                temp_grad_list=[]
                for p in self.model.parameters():
                    temp_grad_list.append(torch.flatten(p.grad.clone()))
                    p.grad *= 0.
                cost = torch.cat(temp_grad_list,dim=0)
                dummy_optimizer.zero_grad()

                # TD(lambda) update
                e_vec = gamma*lam*e_vec + feature(next_ob)
                a_mat += torch.mm(e_vec.unsqueeze(1), (feature(next_ob)-gamma*feature(ob_torch)).unsqueeze(0))
                b_vec += gamma*torch.mm(e_vec.unsqueeze(1), cost.unsqueeze(0))
                c_mat += torch.mm(feature(ob_torch).unsqueeze(1), feature(ob_torch).unsqueeze(0))
                #c_mat += torch.mm(e_vec.unsqueeze(1), feature(next_ob).unsqueeze(0))
                phi_sum += feature(next_ob)

                ob = next_ob.numpy()
                t += 1
                if np.random.rand()>gamma:
                    break
                if termination(ob) is True:
                    break
            return a_mat, b_vec, t, c_mat, phi_sum


        a_mat1=0.
        b_vec1=0.
        t_len1=0
        c_mat1=0.
        phi_ave=0.
        prev_lin_param=0.
        for i in range(num_iter):
            temp_a_mat, temp_b_vec, temp_t, temp_c_mat, temp_phi_sum = get_sim_one_episode()
            a_mat1 = (a_mat1*t_len1 + temp_a_mat) / (t_len1 + temp_t)
            b_vec1 = (b_vec1*t_len1 + temp_b_vec) / (t_len1 + temp_t)
            c_mat1 = (c_mat1*t_len1 + temp_c_mat) / (t_len1 + temp_t)
            phi_ave = (phi_ave*t_len1 + temp_phi_sum) / (t_len1 + temp_t)
            t_len1 += temp_t

            if ((i+1)%100)==0:
                lin_param = torch.mm(torch.inverse(a_mat1), b_vec1) # ordinary lstd
                AtCA=torch.mm(a_mat1.T,torch.mm(torch.inverse(c_mat1),a_mat1)) # lstd with constraint
                AtCb=torch.mm(a_mat1.T,torch.mm(torch.inverse(c_mat1),b_vec1)) # lstd with constraint
                lin_param2 = torch.mm(torch.inverse(AtCA+penalty*torch.mm(phi_ave.unsqueeze(1),phi_ave.unsqueeze(0))), AtCb) # lstd with constraint

                print("[lstd]",i+1,"param_diff_max",torch.max(torch.abs(lin_param2-prev_lin_param)))
                prev_lin_param = lin_param2
                #print(torch.mm(phi_ave.unsqueeze(0),lin_param))  # be 0 in expectation in theory
                #print(torch.mm(phi_ave.unsqueeze(0),lin_param2)) # be 0 in expectation in theory

        self.lin_param = lin_param2


    def get_grad_vector(self, input_data, output_data, data_weight, b_hat=1.): # averaged per data
        ob_dim=output_data.shape[1]

        temp_input_data  = torch.from_numpy(input_data.astype(np.float32)).clone()
        temp_output_data = torch.from_numpy(output_data.astype(np.float32)).clone()

        temp_list=[ feature(temp_input_data[i,:ob_dim]).unsqueeze(0) for i in range(input_data.shape[0])]
        grad_batch = torch.mm(torch.cat(temp_list), self.lin_param)

        with torch.no_grad():
            mu, logvar = self.model(temp_input_data)
            invvar = torch.exp(-logvar)
            unweighted_nll = 0.5 * (torch.unsqueeze(torch.sum( (invvar*((mu-temp_output_data)**2) + logvar ),dim=1),1)   + np.log(2.*np.pi)*ob_dim )
            #print("unweighted_nll.shape",unweighted_nll.shape)
        #reward_fn=None
        #if reward_fn is not None:
        #    #unweighted_nll *= b_hat
        #    for i in range(input_data.shape[0]):
        #        unweighted_nll[i] -= reward_fn(temp_input_data[i])/((1.-self.gamma)*b_hat)

        return torch.sum(unweighted_nll * data_weight * grad_batch, dim=0)/data_weight.sum()


    """
    def get_grad_vector2(self, input_data, output_data, data_weight, unweighted_nll, reward_fn=None, b_hat=1.): # averaged per data
        ob_dim=output_data.shape[1]

        temp_input_data  = torch.from_numpy(input_data.astype(np.float32)).clone()
        temp_output_data = torch.from_numpy(output_data.astype(np.float32)).clone()

        temp_list=[ feature(temp_input_data[i,:ob_dim]).unsqueeze(0) for i in range(input_data.shape[0])]
        grad_batch = torch.mm(torch.cat(temp_list),self.lin_param)

        if reward_fn is not None:
            #unweighted_nll *= b_hat
            for i in range(input_data.shape[0]):
                unweighted_nll[i] -= reward_fn(temp_input_data[i])/((1.-self.gamma)*b_hat)

        return torch.sum(unweighted_nll * data_weight * grad_batch, dim=0)/data_weight.sum()
    """
