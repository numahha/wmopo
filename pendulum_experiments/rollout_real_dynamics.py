import numpy as np
import torch
import gym, custom_gym


def rollout(env_fn, 
            seed=1,
            gamma=1.,
            num_test_episodes=100):

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = env_fn()

    ac = torch.load('torch_policy.pt')
    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    ep_ret_list=[]
    ob_list=[]
    for j in range(num_test_episodes):
        o, d, ep_ret, ep_len = env.reset(), False, 0, 0
        env.env.state = np.random.randn(2)*1. + np.array([np.pi, 0])
        o = env.env.state[:]
        init_o=o
        temp_gamma=1
        while not d:
            ob_list.append(o)
            env.render()
            a = get_action(o)
            o, r, d, _ = env.step(a)
            #print(o)
            ep_ret += temp_gamma*r
            ep_len += 1
            temp_gamma = temp_gamma*gamma
            #if np.random.rand()>0.99:
            #    break
        #print(ep_ret,"init_ob =",init_o,"end_ob =",o)
        ep_ret_list.append(ep_ret)
        print(j,ep_ret)
    ep_ret_list = np.array(ep_ret_list)
    print("gamma",gamma)
    print("mean",ep_ret_list.mean())
    print("std",ep_ret_list.std())
    ob_list= np.array(ob_list)
    import matplotlib.pyplot as plt
    from exp_common import ExpCommon
    common=ExpCommon()
    obac_data    = common.obac_data
    offline_data = np.loadtxt('np_obac.csv',delimiter=',')
    #temp_weight = np.loadtxt('temp_mle_dynamics_model_iw.csv',delimiter=',')
    #temp_weight = np.loadtxt('temp_mle_e_step5_iw.csv',delimiter=',')
    #plt.scatter(x=obac_data[:,0], y=obac_data[:,1], c=np.log10(temp_weight), cmap='jet')
    #plt.scatter(x=obac_data[index,0], y=obac_data[index,1], c=temp_weight[index,0], cmap='jet')
    #cb=plt.colorbar()
    #cb.remove()
    plt.scatter(obac_data[:,0], obac_data[:,1], facecolors='none', edgecolors='k')


    plt.plot(ob_list[:1000,0], ob_list[:1000,1],"r*")
    plt.xlim([-4.1*np.pi,4.1*np.pi])
    plt.ylim([-15,15])
    plt.xlabel('Angle', fontsize=18)
    plt.ylabel('Angular velocity', fontsize=18)
    plt.savefig("fig_1_a.pdf")
    plt.close()    

if __name__ == '__main__':

    from env_def import EnvDef
    envdef=EnvDef()

    rollout(lambda : gym.make(envdef.env_name), 
        seed=0,
        gamma=envdef.gamma)
