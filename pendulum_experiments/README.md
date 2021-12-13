# Pendulum

## policy evaluation (unweighted maximum likelihood estimation)
```
$ python mle_e_step.py --unweighted_mle True
```

## policy evaluation (weighted ERM, full version)
```
$ python mle_e_step.py
```

## policy evaluation (weighted ERM, simplified version)
```
$ python mle_e_step.py --skip_grad True
```

## applying to real environment
```
$ python rollout_real_dynamics.py
```

## applying to simulation environment
```
$ python rollout_model.py
```

## generate offline data
```
$ python generate_offline_data.py
```


