import gym, custom_gym
import numpy as np
from env_def import EnvDef
envdef=EnvDef()
env = gym.make(envdef.env_name)
o = env.reset()
th_list=[]
ob_list=[]
ac_list=[]
diff_ob_list=[]

#import torch
#ac = torch.load('pendulum_policy.pt')
#def get_action(o, deterministic=False):
#    return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

def get_random_action(o):
    return env.action_space.sample()

#pol = get_action
pol = get_random_action

for _ in range(1000):
    #env.render()
    a = pol(o)

    ob_list.append(o)
    ac_list.append(a)

    nexto, reward, done, info = env.step(a)

    diff_ob_list.append(nexto-o)
    #diff_ob_list.append([env.env.state[0] - th, nexto[2]-o[2]]) # only for pendulum
    
    o = nexto
    
    if done:
        #pol = get_random_action
        o = env.reset()

env.close()

import numpy as np
ob_list = np.array(ob_list)
ac_list = np.array(ac_list)
obac_list = np.concatenate((ob_list,ac_list),axis=1)
np.savetxt('np_obac.csv', obac_list, delimiter=',')

diff_ob_list = np.array(diff_ob_list)
np.savetxt('np_diff_ob.csv', diff_ob_list, delimiter=',')

import matplotlib.pyplot as plt
plt.plot(ob_list[:,0],ob_list[:,1])
plt.show()
