import numpy as np
import gym


def term_halfcheetah(obs, act, next_obs):
    return np.zeros((next_obs.shape[0],1), dtype=bool)

def term_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  (height > 0.8) \
                * (height < 2.0) \
                * (angle > -1.0) \
                * (angle < 1.0)
    done = ~not_done
    done = done[:,None]
    return done


def term_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  np.isfinite(next_obs).all(axis=-1) \
                * (np.abs(next_obs[:,1:]) < 100).all(axis=-1) \
                * (height > .7) \
                * (np.abs(angle) < .2)
    done = ~not_done
    done = done[:,None]
    return done



class EnvInitTerm():
    def __init__(self, env_name, seed):
        self.env_name = env_name
        if self.env_name=='halfcheetah':
            self.local_env = gym.make("HalfCheetah-v2")
        if self.env_name=='walker2d':
            self.local_env = gym.make("Walker2d-v2")
        if self.env_name=='hopper':
            self.local_env = gym.make("Hopper-v2")
        self.local_env.seed(seed)

    def set_ob_space(self):
        return self.local_env.observation_space

    def set_ac_space(self):
        return self.local_env.action_space

    def set_init_ob(self):
        return self.local_env.reset


    def set_term(self):
        if self.env_name=='halfcheetah':
            ret = term_halfcheetah
        if self.env_name=='walker2d':
            ret = term_walker2d
        if self.env_name=='hopper':
            ret = term_hopper
        return ret



