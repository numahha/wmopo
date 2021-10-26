import numpy as np
import matplotlib.pyplot as plt
import gym
import d4rl
import copy

from .model_dyna_nn_torch import ModelDynaNNEnsemble
from .model_ratio import ModelRatio
from .model_penalty import ModelPenalty
from .env_init_term import EnvInitTerm



class EStep:

    def __init__(self, env_name, dataset_name, gamma, alpha, seed, path="./"):

        self.alpha = alpha
        self.gamma = gamma
        self.B_dash = 1.e-2
        self.path = path
        self.seed = seed

        self.exp_name = env_name+"-"+dataset_name+'-v2' 
        env = gym.make(self.exp_name)
        
        dataset = env.get_dataset()
        obs = dataset['observations']
        act = dataset['actions']
        next_obs = dataset['next_observations']
        rew = np.expand_dims(dataset['rewards'], 1)

        self.inputs  = np.concatenate((obs, act), axis=-1)
        self.targets = np.concatenate((rew, next_obs-obs), axis=-1)

        self.dr_weights = np.ones((self.targets.shape[0],1))
        self.previous_weights_for_training = np.ones((self.targets.shape[0],1))

        self.envinitterm = EnvInitTerm(env_name, seed)
        self.init_ob_fn = self.envinitterm.set_init_ob()
        self.term_fn    = self.envinitterm.set_term()
        self.observation_space = self.envinitterm.set_ob_space()
        self.action_space      = self.envinitterm.set_ac_space()

        self.rew_abs_max = max(np.abs(rew.min()),np.abs(rew.max()))

        self.hidden_num=256
        self.batch_size=512
        self.hold_out_ratio = 0.1
        self.early_stop_num=5

        self.model_dyna    = ModelDynaNNEnsemble(self.inputs, self.targets, 
                                                 self.seed, self.hidden_num, self.batch_size,
                                                 self.hold_out_ratio, self.early_stop_num)
        self.model_ratio   = ModelRatio( self.inputs.shape[1], self.hold_out_ratio, self.early_stop_num )
        self.model_penalty = ModelPenalty( self.inputs.shape[1], self.hold_out_ratio, self.early_stop_num )

        self.b_coeff  = 0. # dummy value. to be overwritten later.
        self.loss_min = 0. # dummy value. to be overwritten later.

    def model_save(self, iter_num):
        if iter_num==0:
            model_name = "model_dyna_iter"+str(iter_num)
        else:
            model_name = "model_dyna_iter"+str(iter_num)+"_alpha"+str(self.alpha)
        self.model_dyna.save_model(self.path+model_name)

        if iter_num==0:
            model_name = "model_ratio_iter"+str(iter_num)
        else:
            model_name = "model_ratio_iter"+str(iter_num)+"_alpha"+str(self.alpha)
        self.model_ratio.save_model(self.path+model_name)
        if iter_num==0:
            model_name = "model_penalty_iter"+str(iter_num)
        else:
            model_name = "model_penalty_iter"+str(iter_num)+"_alpha"+str(self.alpha)
        self.model_penalty.save_model(self.path+model_name)

    def model_load(self, iter_num):

        if iter_num==0:
            model_name = "model_dyna_iter"+str(iter_num)
        else:
            model_name = "model_dyna_iter"+str(iter_num)+"_alpha"+str(self.alpha)
        self.model_dyna.load_model(self.path+model_name)
        if iter_num==0:
            model_name = "model_ratio_iter"+str(iter_num)
        else:
            model_name = "model_ratio_iter"+str(iter_num)+"_alpha"+str(self.alpha)
        self.model_ratio.load_model(self.path+model_name)
        if iter_num==0:
            model_name = "model_penalty_iter"+str(iter_num)
        else:
            model_name = "model_penalty_iter"+str(iter_num)+"_alpha"+str(self.alpha)
        self.model_penalty.load_model(self.path+model_name)
        splitnum=100        
        train_inputs_list = np.array_split(self.inputs, splitnum)
        dr_weights = self.model_ratio.predict(train_inputs_list[0])
        for i in range(1,splitnum):
            temp_dr_weights = self.model_ratio.predict(train_inputs_list[i])
            dr_weights = np.concatenate([dr_weights, temp_dr_weights], 0)
        self.dr_weights = dr_weights

    def update(self, get_action, iter_num):
        print("--------------")
        print("Start E-step !")
        print("--------------")
        print("iter_num",iter_num)
        if (iter_num==0) or (self.alpha<1.e-6):
            try:
                self.model_load(0)
                pointwise_losses, pred_pointwise_losses = self.train_model_penalty(train_flag=False)
                if iter_num>0:
                    self.train_model_ratio(get_action)
                self.load_flag=True
                print("load dyna model")

            except:
                print("learn dyna model")
                self.train_model_dyna()
                self.train_model_ratio(get_action)
                pointwise_losses, pred_pointwise_losses = self.train_model_penalty()            
                self.load_flag=False

            f = open(self.path+"EstepInfo_iter"+str(iter_num)+".txt", 'w')
        else:
            try:
                self.model_load(iter_num)
                pointwise_losses, pred_pointwise_losses = self.train_model_penalty(train_flag=False)
                self.load_flag=True
                print("load dyna model")
            except:
                print("learn dyna model")
                self.model_load(iter_num-1)
                self.train_model_ratio(get_action)
                self.train_model_dyna()

                self.train_model_ratio(get_action)
                pointwise_losses, pred_pointwise_losses = self.train_model_penalty()
                self.load_flag=False

            f = open(self.path+"EstepInfo_iter"+str(iter_num)+"_alpha"+str(self.alpha)+".txt", 'w')

        print("shape",pointwise_losses.shape, self.dr_weights[:,0].shape)
        sqrt_L_minus_h = np.sqrt( np.sum(self.dr_weights[:,0]*pointwise_losses))
        self.b_coeff = 0.5 * self.B_dash / np.sqrt( sqrt_L_minus_h ) 

        f.write("Ensemble NNs\n")
        f.write("seed = "+str(self.seed)+"\n")
        f.write("load_flag ="+str(self.load_flag)+"\n")
        f.write("gamma ="+str(self.gamma)+"\n")
        f.write("B_dash ="+str(self.B_dash)+"\n")
        f.write("hidden_num = "+str(self.hidden_num)+"\n")
        f.write("batch_size = "+str(self.batch_size)+"\n")
        f.write("early_stop_num = "+str(self.early_stop_num)+"\n")
        f.write("hold_out_ratio = "+str(self.hold_out_ratio)+"\n")
        f.write("sqrt_L_minus_h = "+str(sqrt_L_minus_h)+"\n")
        f.write("b_coeff = "+str(self.b_coeff)+"\n")
        f.write("loss_min = "+str(self.loss_min)+"\n")
        f.write("\n\nmaxlogvar = "+str(self.model_dyna.model.max_logvar.cpu().detach().numpy())+"\n")
        f.close()


        print("---------------")
        print("Finish E-step !")
        print("---------------")
        self.model_save(iter_num)

        return pointwise_losses, pred_pointwise_losses, self.dr_weights, self.load_flag


    def train_model_dyna(self, reset_flag=False):
        print(self.exp_name, "alpha =",self.alpha)
        print("min_weight",self.dr_weights.min(), "max_weight",self.dr_weights.max())
        temp_weights = self.dr_weights**self.alpha
        temp_weights = temp_weights *temp_weights.shape[0] / temp_weights.sum()
        self.weights_for_training = temp_weights
        self.model_dyna.train(copy.deepcopy(self.inputs), copy.deepcopy(self.targets), temp_weights, reset_flag=reset_flag)


    def train_model_ratio(self, policy, reset_flag=False):

        self.dr_weights=0.
        temp_inputs = copy.deepcopy(self.inputs)
        data_num = self.targets.shape[0]
        sim_samples = self.obac_simulation(policy, data_num)

        self.model_ratio.train(temp_inputs, sim_samples, reset_flag=reset_flag)

        splitnum=100
        train_inputs_list = np.array_split(self.inputs, splitnum)
        dr_weights = self.model_ratio.predict(train_inputs_list[0])
        for i in range(1,splitnum):
            temp_dr_weights = self.model_ratio.predict(train_inputs_list[i])
            dr_weights = np.concatenate([dr_weights, temp_dr_weights], 0)
        self.dr_weights = dr_weights

    def train_model_penalty(self, train_flag=True):
        splitnum=100        
        temp_inputs_list  = np.array_split(self.inputs, splitnum)
        temp_targets_list = np.array_split(self.targets, splitnum)
        pointwise_losses = self.model_dyna.get_pointwise_loss(temp_inputs_list[0], temp_targets_list[0])
        for i in range(1,splitnum):
            temp_pointwise_losses = self.model_dyna.get_pointwise_loss(temp_inputs_list[i], temp_targets_list[i])
            pointwise_losses = np.concatenate([pointwise_losses, temp_pointwise_losses], 0)
        self.loss_min = pointwise_losses.min()
        pointwise_losses -= pointwise_losses.min()

        if train_flag:
            self.model_penalty.train(copy.deepcopy(self.inputs), pointwise_losses)

        pred_pointwise_losses = self.model_penalty.predict(temp_inputs_list[0])
        for i in range(1,splitnum):
            temp_pred_pointwise_losses = self.model_penalty.predict(temp_inputs_list[i])
            pred_pointwise_losses = np.concatenate([pred_pointwise_losses, temp_pred_pointwise_losses], 0)
        return pointwise_losses, pred_pointwise_losses


    def obac_simulation(self, policy, data_num, parallel_num=200):

        def local_policy(o):
            return np.clip(policy(o), self.action_space.low, self.action_space.high)

        ob = np.array([ self.init_ob_fn() for i in range(parallel_num)])
        ac = local_policy(ob)
        ob_store = copy.deepcopy(ob)
        ac_store = copy.deepcopy(ac)
        total_ob_store = None
        total_ac_store = None
        while True:
            overflowFlag=False
            while True:
                ob, rew, term, info = self.multiple_step(ob, ac)
                nonterm_mask = ~term.squeeze(-1)
                ob = ob[nonterm_mask]
                temp_rand = np.random.rand(ob.shape[0])
                ob = ob[np.where(temp_rand<self.gamma)]
                if ob.shape[0] == 0:
                    break
                if np.count_nonzero(np.isnan(ob))>0:
                    overflowFlag=True
                    break
                ac = local_policy(ob)
                if np.count_nonzero(np.isnan(ac))>0:
                    overflowFlag=True
                    break
                ob_store = np.concatenate([ob_store,ob])
                ac_store = np.concatenate([ac_store,ac])

            if overflowFlag is False:
                if total_ob_store is None:
                    total_ob_store = copy.deepcopy(ob_store)
                    total_ac_store = copy.deepcopy(ac_store)
                else:
                    total_ob_store = np.concatenate([total_ob_store,ob_store])
                    total_ac_store = np.concatenate([total_ac_store,ac_store])

            if total_ob_store is not None:
                print("\rtotal_ob_store_num",total_ob_store.shape[0],"overflowFlag",overflowFlag,end="\r")
                if total_ob_store.shape[0]>data_num:
                    break

            ob = np.array([ self.init_ob_fn() for i in range(parallel_num)])
            ac = local_policy(ob)
            ob_store = copy.deepcopy(ob)
            ac_store = copy.deepcopy(ac)

        ret_data = np.concatenate([total_ob_store,total_ac_store],axis=1)
        np.random.shuffle(ret_data)
        return ret_data[:int(data_num)]

    def penalty_fn(self, obac):
        return self.b_coeff*self.model_penalty.predict(obac)


    def multiple_step(self, obs, act):
        inputs = np.concatenate((obs, act), axis=-1)
        rewards, next_obs = self.model_dyna.predict(inputs)
        terminals = self.term_fn(obs, act, next_obs)


        scl=10.
        for i in range(next_obs.shape[0]):
            if ((np.count_nonzero(np.isnan(next_obs[i]))+np.count_nonzero(np.isnan(rewards[i]))) > 0) or (np.abs(next_obs[i]).max()>1.e10) or (np.abs(rewards[i]).max()>scl*self.rew_abs_max):
                terminals[i]=np.ones((1,), dtype=bool)
                if np.isnan(rewards[i]):
                    rewards[i] = -scl*self.rew_abs_max*np.ones((1,))
                next_obs[i] = obs[i]

        unpenalized_rewards = rewards
        penalized_rewards = rewards - self.penalty_fn(inputs)


        penalized_rewards = np.clip(penalized_rewards, -scl*self.rew_abs_max, scl*self.rew_abs_max)

        info = {'unpenalized_rewards': unpenalized_rewards, 'penalized_rewards': penalized_rewards}
        return next_obs, penalized_rewards, terminals, info


