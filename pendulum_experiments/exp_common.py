from regression import DynamicsRegression
from nll_estimation import NLLRegression
import torch
import numpy as np


from env_def import EnvDef

def default_c_hat(sa):
    #print("hello")
    return 0.

class ExpCommon():

    def __init__(self,m_step_flag=False,
                 hidden_unit_num=8,
                 B_dash=1):


        self.obac_data    = np.loadtxt('np_obac.csv',delimiter=',')
        self.diff_ob_data = np.loadtxt('np_diff_ob.csv',delimiter=',')
           
        # construct model
        self.dynamics_model = DynamicsRegression(self.obac_data, self.diff_ob_data, hidden_unit_num=hidden_unit_num, B_dash=B_dash)

        envdef = EnvDef()
        self.gamma = envdef.gamma
        self.termination = envdef.termination
        self.reward_fn = envdef.reward_fn
        self.reset = envdef.reset
        self.env_name = envdef.env_name

        if m_step_flag is False:
            self.c_hat=default_c_hat
        else:
            self.nllmodel=NLLRegression(self.obac_data, np.loadtxt('temp_unweighted_nll.csv',delimiter=','))
            self.nllmodel.train_model()
            self.c_hat=self.nllmodel.pred

    def custom_reward_for_optimization(self, sa):
        return self.reward_fn(sa) - (1.-self.gamma)*self.b_hat*self.c_hat(sa)
        #return - (1. - np.exp(-1.*(sa[0]**2)))

    def reset2(self):
        return self.reset()

    def wrap(self,local_envfn):
        self.dynamics_model.load_model()
        local_envfn.one_step = self.dynamics_model.sim_next_ob
        self.b_hat=self.dynamics_model.get_b_hat()
        local_envfn.reset = self.reset2
        print("(1.-gamma)*b_hat =",(1.-self.gamma)*self.b_hat)
        local_envfn.reward = self.custom_reward_for_optimization

