import argparse
import numpy as np
import torch
import time
from model_env.e_step import EStep
from sac.m_step_parallel import MStep
import os


if __name__ == '__main__':    

    parser = argparse.ArgumentParser()

    #parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--env', type=str, default='walker2d')
    #parser.add_argument('--env', type=str, default='hopper')

    #parser.add_argument('--dataset', type=str, default='random')
    #parser.add_argument('--dataset', type=str, default='medium')
    #parser.add_argument('--dataset', type=str, default='medium-replay')
    parser.add_argument('--dataset', type=str, default='medium-expert')

    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--alpha', type=float, default=0.1) # this is for regularization coefficient for density ratio, NOT entropy for SAC.
    #parser.add_argument('--seed', '-s', type=int, default=np.random.randint(0,10000))
    parser.add_argument('--seed', '-s', type=int, default=4)
    parser.add_argument('--m_lr', type=float, default=0.001) # learning rate of SAC
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    path = 'data/'+args.env+"-"+args.dataset+'-v2/seed'+str(args.seed)+'/'
    print("path =",path)

    if not os.path.exists(path):
            os.makedirs(path)

    m_step = MStep( env_name=args.env, 
                    dataset_name=args.dataset, 
                    gamma=args.gamma,
                    seed=args.seed,
                    lr=args.m_lr,
                    path=path)

    e_step = EStep( env_name=args.env, 
                    dataset_name=args.dataset, 
                    gamma=args.gamma, 
                    alpha=args.alpha,
                    seed=args.seed,
                    path=path)

    start_time = time.time()

    print("args.alpha =",args.alpha)
    
    load_flag = m_step.model_load(0, args.alpha)
    if load_flag is False:
        i=0
        pointwise_losses, pred_pointwise_losses, _, model_load_flag = e_step.update( m_step.get_action, i )

        if model_load_flag:
            m_stats = m_step.update( model_step=e_step.multiple_step, 
                                     get_pointwise_loss=e_step.model_dyna.get_pointwise_loss,
                                     b_coeff=e_step.b_coeff )
            np.savetxt(path+"progress_ite"+str(i)+".csv",m_stats)
            m_step.model_save(i, args.alpha)

    else:
        load_flag = m_step.model_load(1, args.alpha)
        if load_flag is False:
            i=1
        else:
            i=2
        print("\nSeed", args.seed, "Iteration", i, "start", "total sec", int(time.time()-start_time), "\n")
        pointwise_losses, pred_pointwise_losses, dr_weights, model_load_flag = e_step.update( m_step.get_action, i )

        if model_load_flag:

            print("\nSeed",args.seed,"Iteration",i, "half", "total sec",int(time.time()-start_time), "\n")
            m_stats = m_step.update( model_step=e_step.multiple_step, 
                                     get_pointwise_loss=e_step.model_dyna.get_pointwise_loss,
                                     b_coeff=e_step.b_coeff,
                                     start_steps=0 )
            np.savetxt(path+"progress_alpha"+str(args.alpha)+"_iter"+str(i)+".csv",m_stats)

            m_step.model_save(i, args.alpha)


    print("\nSeed",args.seed,"Iteration i",i, "total sec",int(time.time()-start_time))
    print("Complete")

