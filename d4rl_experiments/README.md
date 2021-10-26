# D4RL MuJoCo Benchmark

## Run
```
python main.py --env "walker2d" --dataset "medium-expert"                # ERM    in 1st iter in Alg 3 (common to alpha=0 and alpha=0.2)
python main.py --env "walker2d" --dataset "medium-expert"                # M-step in 1st iter in Alg 3 (common to alpha=0 and alpha=0.2)
python main.py --env "walker2d" --dataset "medium-expert" --alpha 0.2    # E-step in 2nd iter in Alg 3
python main.py --env "walker2d" --dataset "medium-expert" --alpha 0.2    # M-step in 2nd iter in Alg 3
python main.py --env "walker2d" --dataset "medium-expert" --alpha 0.0    # M-step in 2nd iter in Alg 3
```

