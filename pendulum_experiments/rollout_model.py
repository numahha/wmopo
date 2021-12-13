import numpy as np
import torch
torch.manual_seed(1)
np.random.seed(1)

def get_sim_one_episode(sim_model, policy, init_dist, termination, gamma):
    ob = init_dist()
    obac_list = []
    ret = 0.
    temp_gamma=1.
    while 1:
        ac = policy(ob)
        obac_list.append(np.concatenate((ob,ac),axis=0))
        rew = -(1. - np.exp(-1.*(ob[0]**2)))
        ret += temp_gamma*rew
        temp_gamma *= gamma
        ob = sim_model(ob,ac)
        #if np.random.rand()>gamma:
        #    break
        #if termination(ob) is True:
        #    break
        if len(obac_list)==200:
            break
    return np.array(obac_list), ret


def get_sim_data(sim_model, policy, init_dist, termination, gamma):
    sim_data, sim_ret = get_sim_one_episode(sim_model, policy, init_dist, termination, gamma)
    sim_ret_list = [sim_ret]
    for i in range(99):
        temp_data, temp_ret = get_sim_one_episode(sim_model, policy, init_dist, termination, gamma)
        sim_data = np.concatenate([sim_data, temp_data], 0)
        sim_ret_list.append(temp_ret)
    return sim_data, sim_ret_list



def rollout_model():

    from exp_common import ExpCommon
    common=ExpCommon()
    obac_data    = common.obac_data
    diff_ob_data = common.diff_ob_data
    dynamics_model = common.dynamics_model

    policy_torch = torch.load('torch_policy.pt')
    def temp_policy(ob_input, deterministic=False):
        return policy_torch.act(torch.as_tensor(ob_input, dtype=torch.float32),deterministic)

    temp_gamma = common.gamma
    temp_reset = common.reset
    temp_termination = common.termination
    temp_reward = common.reward_fn

    dynamics_model.load_model()

    sim_data_example, sim_ret_example = get_sim_data(dynamics_model.sim_next_ob, temp_policy, temp_reset, temp_termination, temp_gamma)

    import matplotlib.pyplot as plt
    #from matplotlib.colors import LogNorm
    #imshow(data,norm=LogNorm())
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    plt.plot(sim_data_example[:1000,0], sim_data_example[:1000,1],"kx",markersize=4)
    print("mean",np.mean(sim_ret_example), "mean",np.std(sim_ret_example))


    offline_data = np.loadtxt('np_obac.csv',delimiter=',')
    temp_weight = np.loadtxt('temp_mle_dynamics_model_iw.csv',delimiter=',')
    #temp_weight = np.loadtxt('temp_mle_e_step0_iw.csv',delimiter=',')*0. + 1.
    plt.scatter(x=obac_data[:,0], y=obac_data[:,1], c=np.log10(temp_weight), cmap='jet', vmin=-2.5, vmax=2.5)
    #plt.scatter(x=obac_data[index,0], y=obac_data[index,1], c=temp_weight[index,0], cmap='jet')
    cb=plt.colorbar()
    #cb.remove()


    #plt.plot(offline_data[:,0], offline_data[:,1],"o")
    #offline_data = np.loadtxt('np_obac.csv',delimiter=',')

    plt.xlim([-4.1*np.pi,4.1*np.pi])
    plt.ylim([-15,15])
    plt.xlabel('Angle', fontsize=18)
    plt.ylabel('Angular velocity', fontsize=18)
    plt.savefig("fig_1_cd.pdf")
    plt.close()

rollout_model()

