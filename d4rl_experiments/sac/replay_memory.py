import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

import gym
import d4rl

class ReplayMemory2:
    def __init__(self, env_name, dataset_name, get_pointwise_loss, b_coeff):
         
        temp_env = gym.make(env_name+"-"+dataset_name+'-v2')
        dataset = temp_env.get_dataset()
        self.obs_buf = dataset['observations']
        self.obs2_buf = dataset['next_observations']
        self.act_buf = dataset['actions']
        self.rew_buf = dataset['rewards']
        self.rew_abs_max = np.abs(self.rew_buf.min())
        if self.rew_abs_max > np.abs(self.rew_buf.max()):
            self.rew_abs_max = np.abs(self.rew_buf.max())        

        inputs  = np.concatenate((self.obs_buf, self.act_buf), axis=-1)
        targets = np.concatenate((np.expand_dims(self.rew_buf, 1), self.obs2_buf-self.obs_buf), axis=-1)

        splitnum=100

        inputs_list  = np.array_split(inputs, splitnum)
        targets_list = np.array_split(targets, splitnum)
        pointwise_losses = get_pointwise_loss(inputs_list[0], targets_list[0])
        for i in range(1,splitnum):
            temp_pointwise_losses = get_pointwise_loss(inputs_list[i], targets_list[i])
            pointwise_losses = np.concatenate([pointwise_losses, temp_pointwise_losses], 0)
        pointwise_losses -= pointwise_losses.min()
        self.rew_buf -= b_coeff * pointwise_losses
        self.rew_buf = np.clip(self.rew_buf, -10.*self.rew_abs_max, 10.*self.rew_abs_max)

        self.done_buf = - dataset['terminals'].astype(np.float32) + 1.0
        self.size = self.obs_buf.shape[0]
        temp_env.close()

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.obs_buf[idxs], self.act_buf[idxs], self.rew_buf[idxs], self.obs2_buf[idxs], self.done_buf[idxs]

    def __len__(self):
        return self.size
