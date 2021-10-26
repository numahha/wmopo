import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import os
import time
from .sac import SAC
from .replay_memory import ReplayMemory, ReplayMemory2


class MStep:
    def __init__(self, env_name, dataset_name, gamma, seed, lr, path="./"):
        self.env_name = env_name
        self.dataset_name=dataset_name
        self.gamma = gamma
        self.seed  = seed

        if self.env_name=="halfcheetah":
            self.env_name_v2 = "HalfCheetah-v2"
        elif self.env_name=="hopper":
            self.env_name_v2 = "Hopper-v2"
        elif self.env_name=="walker2d":
            self.env_name_v2 = "Walker2d-v2"

        self.exp_name = self.env_name+"-"+self.dataset_name+'-v2' 
        self.num_steps=500000
        self.updates_after=10000
        self.real_ratio=0.5
        self.init_optim_flag=False

        self.parallel_num=1

        self.replay_size = 1000000
        self.batch_size = 512

        self.updates_per_step=1
        self.lr = lr

        env = gym.make(self.env_name_v2)
        self.agent = SAC(env.observation_space.shape[0], env.action_space, self.gamma, self.lr)
        env.close()

        self.path = path
        # output info
        f = open(self.path+"MstepInfo.txt", 'w')
        f.write("seed = "+str(seed)+"\n")
        f.write("gamma = "+str(self.gamma)+"\n")
        f.write("init_optim_flag = "+str(self.init_optim_flag)+"\n")
        f.write("num_steps = "+str(self.num_steps)+"\n")
        f.write("updates_after = "+str(self.updates_after)+"\n")
        f.write("real_ratio = "+str(self.real_ratio)+"\n")
        f.write("replay_size = "+str(self.replay_size)+"\n")
        f.write("batch_size = "+str(self.batch_size)+"\n")
        f.write("updates_per_step = "+str(self.updates_per_step)+"\n")
        f.write("lr = "+str(self.lr)+"\n")
        f.write("parallel_num = "+str(self.parallel_num)+"\n")
        f.close()



    def print_condition(self):
        print("gamma =", self.gamma)
        print("real_ratio =", self.real_ratio)
        print("updates_after", self.updates_after)
        print("updates_per_step", self.updates_per_step)
        print("learning_rate", self.agent.lr)
        print("init_optim_flag",self.init_optim_flag)

    def model_save(self, iter_num=0, dyna_alpha=1.):
        if iter_num==0:
            sac_path = self.path+'sac_init/'
        if iter_num>0:
            sac_path = self.path+'sac_iter'+str(iter_num)+'alpha'+str(dyna_alpha)+'/'
        if not os.path.exists(sac_path):
            os.makedirs(sac_path)
        torch.save(self.agent.policy.state_dict(), sac_path+'actor')
        torch.save(self.agent.critic.state_dict(), sac_path+'critic')
        
    def model_load(self, iter_num=0, dyna_alpha=1.):
        if iter_num==0:
            sac_path = self.path+'sac_init/'
        if iter_num>0:
            sac_path = self.path+'sac_iter'+str(iter_num)+'alpha'+str(dyna_alpha)+'/'
        if not os.path.exists(sac_path):
            return False
        self.agent.policy.load_state_dict(torch.load(sac_path+'actor'))
        self.agent.critic.load_state_dict(torch.load(sac_path+'critic'))
        return True

    def update(self, model_step, get_pointwise_loss, b_coeff, start_steps=10000):

        env = gym.make(self.env_name_v2)
        env.seed(self.seed)
        env.action_space.seed(self.seed)
        memory = ReplayMemory(self.replay_size, self.seed)
        real_memory = ReplayMemory2(self.env_name, self.dataset_name, get_pointwise_loss, b_coeff)
        if self.init_optim_flag:
            self.agent.init_optim()


        print("--------------")
        print("Start M-step !")
        print("--------------")
        self.print_condition()


        progress=[]
        def test_rollout(total_numsteps_):
            test_avg_reward = 0.
            test_episode_steps = 0
            test_episodes = 5
            for _  in range(test_episodes):
                test_state = env.reset()
                test_episode_reward = 0
                test_done = False
                while not test_done:
                    test_action = self.agent.select_action(test_state, evaluate=True)

                    test_next_state, test_reward, test_done, _ = env.step(test_action)
                    test_episode_reward += test_reward
                    test_episode_steps +=1

                    test_state = test_next_state
                test_avg_reward += test_episode_reward
            test_avg_reward /= test_episodes

            test2_avg_reward = 0.
            test2_episode_steps_sum = 0
            test2_episodes = 5
            for _  in range(test2_episodes):
                test2_state = env.reset()
                test2_episode_reward = 0
                test2_done = False
                test2_episode_steps = 0
                while not test2_done:
                    test2_action = self.agent.select_action(test2_state, evaluate=True)

                    test2_next_state, test2_reward, test2_done, _ = model_step(test2_state[None],test2_action[None])
                    test2_next_state, test2_reward, test2_done = test2_next_state[0], test2_reward[0,0], test2_done[0,0]
                    test2_episode_reward += test2_reward

                    test2_state = test2_next_state
                    test2_episode_steps += 1
                    if test2_episode_steps == env._max_episode_steps:
                        test2_done = True
                test2_episode_steps_sum += test2_episode_steps

                test2_avg_reward += test2_episode_reward
            test2_avg_reward /= test2_episodes

            print("Test: RealRet {}, RealStep {}, SimRet {}, SimSteps {}, Time: {} [s], Total_numsteps: {}".format(round(test_avg_reward, 2), 
                                                                                                                   round(test_episode_steps/test_episodes, 2), 
                                                                                                                   round(test2_avg_reward, 2), 
                                                                                                                   round(test2_episode_steps_sum/test2_episodes, 2), 
                                                                                                                   int(time.time()-start_time),
                                                                                                                   total_numsteps_))
            progress.append([total_numsteps,
                             test_avg_reward,
                             test_episode_steps/test_episodes,
                             test2_avg_reward,
                             test2_episode_steps_sum/test2_episodes])
            np.savetxt("temp_sac_progress.csv",np.array(progress))


        # training loop
        total_numsteps = 0
        updates = 0
        start_time = time.time()
        while True:
            episode_steps = 0
            temp_ave = 0.
            done           = np.array([ False for ii in range(self.parallel_num)])
            state          = np.array([ env.reset() for ii in range(self.parallel_num)])

            while (state.shape[0]>0):
                if start_steps > total_numsteps:
                    action = np.array([ env.action_space.sample() for ii in range(state.shape[0])])
                else:
                    action = np.array([self.agent.select_action(state[ii])  for ii in range(state.shape[0])])

                if (len(memory) > self.batch_size) and (total_numsteps>self.updates_after):
                    # Number of updates per step in environment
                    for i in range(self.updates_per_step):
                        for ii in range(state.shape[0]):
                            # Update parameters of all the networks
                            self.agent.update_parameters(memory, real_memory, self.real_ratio, self.batch_size, updates)
                            updates += 1

                next_state, reward, done, _ = model_step(state,action)

                if (np.count_nonzero(np.isnan(action))>0) or (np.count_nonzero(np.isnan(next_state))>0) or (np.count_nonzero(np.isnan(reward))>0):
                    print("np.count_nonzero(np.isnan(state))",np.count_nonzero(np.isnan(state)))
                    print("np.count_nonzero(np.isnan(action))",np.count_nonzero(np.isnan(action)))
                    print("np.count_nonzero(np.isnan(next_state))",np.count_nonzero(np.isnan(next_state)))
                    print("np.count_nonzero(np.isnan(reward))",np.count_nonzero(np.isnan(reward)))
                    
                episode_steps += 1            
                temp_ave += reward.mean()

                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                if episode_steps == env._max_episode_steps:
                    mask = np.array([ 1 for ii in range(state.shape[0])])
                    done = np.array([ [True] for ii in range(state.shape[0])])
                else:
                    mask = np.array([ float(not done[ii,0]) for ii in range(state.shape[0])])

                for ii in range(state.shape[0]):
                    memory.push(state[ii], action[ii], reward[ii,0], next_state[ii], mask[ii]) # Append transition to memory
                    total_numsteps += 1

                    if ((total_numsteps % 10000) == 0) and (total_numsteps >= self.updates_after):
                        test_rollout(total_numsteps)

                nonterm_mask = ~done.squeeze(-1)
                state = next_state[nonterm_mask]
                
            print("Train: sample_num: {}, steps: {}, rew: {}".format(total_numsteps, episode_steps, temp_ave),end="       \r")

            if total_numsteps >= self.num_steps:
                break

        
        print("---------------")
        print("Finish M-step !")
        print("---------------")
        self.print_condition()

        env.close()
        return np.array(progress)


    def test(self, model_step):

        env = gym.make(self.env_name_v2)
        env.seed(self.seed)
        env.action_space.seed(self.seed)

        def test_rollout(total_numsteps_):
            test_avg_reward = 0.
            test_episode_steps = 0
            test_episodes = 5
            for _  in range(test_episodes):
                test_state = env.reset()
                test_episode_reward = 0
                test_done = False
                while not test_done:
                    test_action = self.agent.select_action(test_state, evaluate=True)

                    test_next_state, test_reward, test_done, _ = env.step(test_action)
                    test_episode_reward += test_reward
                    test_episode_steps +=1

                    test_state = test_next_state
                test_avg_reward += test_episode_reward
            test_avg_reward /= test_episodes

            test2_avg_reward = 0.
            test2_episode_steps_sum = 0
            test2_episodes = 5
            for _  in range(test2_episodes):
                test2_state = env.reset()
                test2_episode_reward = 0
                test2_done = False
                test2_episode_steps = 0
                while not test2_done:
                    test2_action = self.agent.select_action(test2_state, evaluate=True)

                    test2_next_state, test2_reward, test2_done, _ = model_step(test2_state[None],test2_action[None])
                    test2_next_state, test2_reward, test2_done = test2_next_state[0], test2_reward[0,0], test2_done[0,0]
                    test2_episode_reward += test2_reward

                    test2_state = test2_next_state
                    test2_episode_steps += 1
                    if test2_episode_steps == env._max_episode_steps:
                        test2_done = True
                test2_episode_steps_sum += test2_episode_steps

                test2_avg_reward += test2_episode_reward
            test2_avg_reward /= test2_episodes

            print("Test: RealRet {}, RealStep {}, SimRet {}, SimSteps {}, Total_numsteps: {}".format(round(test_avg_reward, 2), 
                                                                                                                   round(test_episode_steps/test_episodes, 2), 
                                                                                                                   round(test2_avg_reward, 2), 
                                                                                                                   round(test2_episode_steps_sum/test2_episodes, 2), 
                                                                                                                   total_numsteps_))
        test_rollout(10)

    def get_action(self, o):
        return self.agent.select_action(o)


