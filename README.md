# Weighted model estimation for offline MBRL

This repository is the official implementation of Weighted model estimation for offline model-based reinforcement learning. 

## Requirements

To install requirements:

1. Install basic libaraies using conda:
```
conda create -n wmopo python=3.7
conda activate wmopo
pip install torch==1.7.1 matplotlib scipy gym 
```

2. Install MuJoCo and mujoco-py, see [MuJoCo webpage](http://www.mujoco.org/) and [mujoco-py webpage](https://github.com/openai/mujoco-py).

3. Install D4RL, see [D4RL webpage](https://github.com/rail-berkeley/d4rl).



## Pendulum

### Training

To fit a model using ERM,
```
python mle_e_step.py --unweighted_mle True --seed 1    # ERM
```

To fit a model using Algorithm 1 (full version),
```
python mle_e_step.py --unweighted_mle True --seed 1    # ERM for initial estimate
python mle_e_step.py --seed 1                          # Alg 1
```

To fit a model using Algorithm 2 (simplified version),
```
python mle_e_step.py --unweighted_mle True --seed 1    # ERM for initial estimate
python mle_e_step.py --skip_grad True --seed 1         # Alg 2
```

### Evaluation and Results
The real expected return of target policy is obtained by 
```
python rollout_real_dynamics.py
```
After fitting a model as described above, the simualtion expected return is obtained by 
```
python rollout_model.py
```
These commands will obtain the expected returns and Figs 1 (a,c,d).

Script `pendulum_experiments/plot_fig_1_b.py` will obtain Fig 1 (b).



## D4RL MuJoCo Benchmark
This paper uses two desktop PCs, with GeForce RTX 2060 SUPER and GeForce RTX 2070 SUPER (cuda 10.2 and cudnn 7.6.5).

### Training

To execute a run on walker2d-medium-expert dataset that is discussed in detail in this paper,

```
python main.py --env "walker2d" --dataset "medium-expert" --seed 2                # ERM    in 1st iter in Alg 3 (common to alpha=0 and alpha=0.2)
python main.py --env "walker2d" --dataset "medium-expert" --seed 2                # M-step in 1st iter in Alg 3 (common to alpha=0 and alpha=0.2)
python main.py --env "walker2d" --dataset "medium-expert" --seed 2 --alpha 0.2    # E-step in 2nd iter in Alg 3
python main.py --env "walker2d" --dataset "medium-expert" --seed 2 --alpha 0.2    # M-step in 2nd iter in Alg 3
python main.py --env "walker2d" --dataset "medium-expert" --seed 2 --alpha 0.0    # M-step in 2nd iter in Alg 3
```


### Evaluation and Results

Each file records the curves of real and simulation returns in the M-step in the 2nd iteration, namely, training and evaluation curves.
Each curve is computed every 10000 SAC updates and averaged over 5 episodes.

We obtain Table 1 converting the last values of the real return curves to the normalization scores, after running 5 runs per dataset.
To compute the normalization scores, see [D4RL webpage](https://github.com/rail-berkeley/d4rl).

|dataset|  alpha=0  | alpha=0.2 |
| ----| ---- | ---- |
|HalfCheetah-random       | 48.7 ± 2.8 | 49.1 ± 3.2 |
|HalfCheetah-medium       | 75.7 ± 1.5 | 73.1 ± 5.2 |
|HalfCheetah-medium-replay| 72.1 ± 1.4 | 65.5 ± 6.4 |
|HalfCheetah-medium-expert| 73.9 ± 24.2| 85.7 ± 21.6|
|Hopper-random            | 30.2 ± 4.4 | 32.7 ± 0.5 |
|Hopper-medium            |100.9 ± 2.7 |104.1 ± 1.2 |
|Hopper-medium-replay     | 97.2 ± 10.9|104.0 ± 3.2 |
|Hopper-medium-expert     |109.3 ± 1.1 |104.9 ± 10.1|
|Walker2d-random          | 16.5 ± 6.6 | 18.4 ± 7.6 |
|Walker2d-medium          | 81.7 ± 1.2 | 60.7 ± 29.0|
|Walker2d-medium-replay   | 80.7 ± 3.1 | 82.7 ± 3.3 |
|Walker2d-medium-expert   | 59.5 ± 49.4|108.2 ± 0.5 |

Script `d4rl_experiments/plot_m_stats.py` will obtain Figure 2 (a) using the curves of real and simulation returns.

## License
MIT License.

