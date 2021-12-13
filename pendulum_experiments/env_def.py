import numpy as np

class PendulumEnvDef():
    def __init__(self):
        self.env_name = 'CustomPendulum-v0'
        self.gamma = 0.99

    def reset(self):
        return np.random.randn(2)*.5 + np.array([np.pi, 0])

    def termination(self,o):
        return False

    def reward_fn(self,sa):
        return - (1. - np.exp(-1.*(sa[0]**2)))


EnvDef = PendulumEnvDef
