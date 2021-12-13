import numpy as np
import torch

def get_sim_one_episode(sim_model, policy, init_dist, termination, gamma):
    ob = init_dist()
    obac_list = []
    while 1:
        ac = policy(ob)
        obac_list.append(np.concatenate((ob,ac),axis=0))
        ob = sim_model(ob,ac)
        if np.random.rand()>gamma:
            break
        if termination(ob) is True:
            break
    return np.array(obac_list)


def get_sim_data(sim_model, policy, init_dist, termination, gamma, data_num):
    sim_data = get_sim_one_episode(sim_model, policy, init_dist, termination, gamma)
    sim_num  = sim_data.shape[0]
    len_list=[]
    while (sim_num<data_num):
        temp_data = get_sim_one_episode(sim_model, policy, init_dist, termination, gamma)
        sim_num   = sim_num + temp_data.shape[0]
        sim_data = np.concatenate([sim_data, temp_data], 0)
        len_list.append(temp_data.shape[0])
    np.random.shuffle(sim_data)
    return sim_data[:data_num]



#def Estep(load_model=True, policy_evaluation_only=False, skip_grad=True):
def Estep(load_model=True, skip_grad=True, alpha=1.):
    policy_evaluation_only=True
    loop_num=10
    ensemble_num=1
    ratio_weight_decay=.0001

    from exp_common import ExpCommon
    common=ExpCommon()
    obac_data    = common.obac_data
    diff_ob_data = common.diff_ob_data
    dynamics_model = common.dynamics_model
    import torch
    policy_torch = torch.load('torch_policy.pt')
    def temp_policy(ob_input, deterministic=False):
        return policy_torch.act(torch.as_tensor(ob_input, dtype=torch.float32),deterministic)

    temp_gamma = common.gamma
    temp_reset = common.reset
    temp_termination = common.termination
    temp_reward = common.reward_fn

    if load_model is True:
        dynamics_model.load_model()

    from ratio_estimation import RatioEstimation
    ratio_model = RatioEstimation(obac_data)

    score_list=[]
    for i in range(loop_num):

        # [1] computing weight
        temp_weight = 0.*dynamics_model.get_data_weight()
        for j in range(ensemble_num):

            sim_data_example = get_sim_data(dynamics_model.sim_next_ob, temp_policy, temp_reset, temp_termination, temp_gamma, int(1e3))

            # weight given by density ratio estimation
            ratio_model.load_sim_data(sim_data_example)
            ratio_model.train_model(weight_decay=ratio_weight_decay)
            temp_weight = (j*temp_weight + ratio_model.output_weight())/(j+1)
            print("temp_weight_sum =",ratio_model.output_weight().sum())
            print("temp_weight_min =",ratio_model.output_weight().min())
            print("temp_weight_max =",ratio_model.output_weight().max())
            #"""
            import matplotlib.pyplot as plt
            plt.plot(sim_data_example[:,0], sim_data_example[:,1],"x")
            plt.xlim([-4.1*np.pi,4.1*np.pi])
            plt.ylim([-15,15])
            plt.savefig("fig_ob_alpha"+str(alpha)+"_iter"+str(i)+".pdf")
            plt.close()
            index=temp_weight[:,0]<10
            index2=temp_weight[:,0]>10
            plt.scatter(x=obac_data[index,0], y=obac_data[index,1], c=temp_weight[index,0], cmap='jet')
            plt.colorbar()
            plt.plot(obac_data[index2,0], obac_data[index2,1],"s")

            plt.xlim([-4.1*np.pi,4.1*np.pi])
            plt.ylim([-15,15])
            plt.savefig("fig_ob2_alpha"+str(alpha)+"_iter"+str(i)+".pdf")
            plt.close()
            #"""

        # [2] computing loss function for evaluation
        dynamics_model.load_data_weight(temp_weight)
        #if policy_evaluation_only is False:
        #    temp_score = - dynamics_model.loss2(temp_reward, temp_gamma)
        #    b_hat = dynamics_model.get_b_hat()
        #    print("b_hat",b_hat)
        #else:
        #    temp_score = - dynamics_model.loss()
        temp_score = - dynamics_model.loss()
        score_list.append(temp_score)
        print("score (larger is better) =",np.array(score_list))
        dynamics_model.save_model("temp_mle_e_step"+str(i))


        #actual_weight = alpha * temp_weight + (1.-alpha)
        actual_weight = temp_weight**alpha
        dynamics_model.load_data_weight(actual_weight)


        # [3] optimizing regression-model-parameter
        if skip_grad is False:
            from grad_lsd import GradLSD
            grad_lsd = GradLSD(dynamics_model=dynamics_model,
                               policy=temp_policy, init_dist=temp_reset, termination=temp_termination, gamma=temp_gamma)
            #if policy_evaluation_only is False:
            #    grad_vector=grad_lsd.get_grad_vector(obac_data, diff_ob_data, actual_weight, reward_fn=temp_reward, b_hat=b_hat)
            #else:
            #    grad_vector=grad_lsd.get_grad_vector(obac_data, diff_ob_data, actual_weight)
            grad_vector=grad_lsd.get_grad_vector(obac_data, diff_ob_data, actual_weight)
            print("grad_vector=",grad_vector)

            dynamics_model.train_model(num_iter=10000,lr=5e-4, grad_vector=grad_vector)
        else:
            dynamics_model.train_model(num_iter=10000,lr=5e-4)


    # last score evaluation
    temp_weight = 0.*dynamics_model.get_data_weight()
    for j in range(ensemble_num):

        sim_data_example = get_sim_data(dynamics_model.sim_next_ob, temp_policy, temp_reset, temp_termination, temp_gamma, int(1e3))
        ratio_model.load_sim_data(sim_data_example)
        ratio_model.train_model(weight_decay=ratio_weight_decay)
        temp_weight = (j*temp_weight + ratio_model.output_weight())/(j+1)
        print("temp_weight_sum =",ratio_model.output_weight().sum())
        print("temp_weight_min =",ratio_model.output_weight().min())
        print("temp_weight_max =",ratio_model.output_weight().max())

        import matplotlib.pyplot as plt
        plt.plot(sim_data_example[:,0], sim_data_example[:,1],"x")
        plt.xlim([-4.1*np.pi,4.1*np.pi])
        plt.ylim([-15,15])
        plt.savefig("fig_ob_alpha"+str(alpha)+"_iter"+str(i+1)+".pdf")
        plt.close()
        index=temp_weight[:,0]<10
        index2=temp_weight[:,0]>10
        plt.scatter(x=obac_data[index,0], y=obac_data[index,1], c=temp_weight[index,0], cmap='jet')
        plt.colorbar()
        plt.plot(obac_data[index2,0], obac_data[index2,1],"s")

        plt.xlim([-4.1*np.pi,4.1*np.pi])
        plt.ylim([-15,15])
        plt.savefig("fig_ob2_alpha"+str(alpha)+"_iter"+str(i+1)+".pdf")
        plt.close()

    # computing loss function for evaluation
    dynamics_model.load_data_weight(temp_weight)
    #if policy_evaluation_only is False:
    #    temp_score = - dynamics_model.loss2(temp_reward, temp_gamma)
    #else:
    #    temp_score = - dynamics_model.loss()
    temp_score = - dynamics_model.loss()
    score_list.append(temp_score)
    print("score (larger is better) =",np.array(score_list))
    dynamics_model.save_model("temp_mle_e_step"+str(i+1))
    dynamics_model.load_model("temp_mle_e_step"+str(np.argmax(np.array(score_list))))
    dynamics_model.save_model()

    np.savetxt('temp_mle_e_step_score_alpha'+str(alpha)+'_skipgrad'+str(skip_grad)+'.csv',np.array(score_list),delimiter=',')
    #dynamics_model.save_model()


def ordinary_model_fitting():
    print("MLE ordinary model fitting")
    from exp_common import ExpCommon
    common=ExpCommon()
    obac_data    = common.obac_data
    diff_ob_data = common.diff_ob_data
    dynamics_model = common.dynamics_model

    temp_weight = 0.*dynamics_model.get_data_weight() + 1
    dynamics_model.load_data_weight(temp_weight)
    #dynamics_model.load_model()
    dynamics_model.train_model(num_iter=50000,lr=5e-4,param_update_penalty=0.)

    dynamics_model.save_model()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--unweighted_mle', type=bool, default=False)
    parser.add_argument('--skip_grad', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.unweighted_mle is False:
        print("alpha =",args.alpha)
        print("skip_grad =",args.skip_grad)
        Estep(skip_grad=args.skip_grad, alpha=args.alpha)
    else:
        print("unweighted maximum likelihood estimation")
        ordinary_model_fitting()

