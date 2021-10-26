import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from .utils import soft_update, hard_update
from .model import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, num_inputs, action_space, gamma, lr):

        self.gamma = gamma
        self.tau = 0.005
        self.alpha = .2
        self.lr = lr

        self.policy_type = "Gaussian"
        self.target_update_interval = 1
        self.automatic_entropy_tuning = False

        self.cuda=True

        self.hidden_size=256
        self.action_space = action_space

        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], self.hidden_size).to(device=self.device)
        #self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], self.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                #self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], self.hidden_size, action_space).to(self.device)
            #self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], self.hidden_size, action_space).to(self.device)
            #self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        self.init_optim()

    def init_optim(self):
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
            self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
            if self.automatic_entropy_tuning is True:
                self.alpha_optim = Adam([self.log_alpha], lr=self.lr)


    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        #return action.detach().cpu().numpy()[0]
        return np.clip(action.detach().cpu().numpy()[0], self.action_space.low, self.action_space.high)

    def update_parameters(self, memory1, memory2, memory2_ratio, batch_size, updates, update_actor=True):
        # Sample a batch from memory
        batch_size_1 = int((1.-memory2_ratio)*batch_size)
        batch_size_2 = batch_size - batch_size_1
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory1.sample(batch_size=batch_size_1)
        if batch_size_2>0:
            state_batch_2, action_batch_2, reward_batch_2, next_state_batch_2, mask_batch_2 = memory2.sample(batch_size=batch_size_2)
            state_batch = np.concatenate((state_batch, state_batch_2), axis=0)
            action_batch = np.concatenate((action_batch, action_batch_2), axis=0)
            reward_batch = np.concatenate((reward_batch, reward_batch_2), axis=0)
            next_state_batch = np.concatenate((next_state_batch, next_state_batch_2), axis=0)
            mask_batch = np.concatenate((mask_batch, mask_batch_2), axis=0)
        #print("shape",state_batch.shape, action_batch.shape, reward_batch.shape, next_state_batch.shape, mask_batch.shape)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        if update_actor:
            pi, log_pi, _ = self.policy.sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        #return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

