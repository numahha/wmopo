import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path



self_max_speed = 15
self_max_torque = 2.5
self_dt = .1
self_g = 10.0
self_m = 1.
self_l = 1.


def delault_one_step(ob, ac):

    th=ob[0]
    thdot=ob[1]
    #newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
    newthdot = thdot + (-3 * self_g / (2 * self_l) * np.sin(th + np.pi) -0.01*(3. / (self_m * self_l ** 2))*thdot + 3. / (self_m * self_l ** 2) * ac[0]) * self_dt
    newth = th + newthdot * self_dt
    #newth = th + thdot * dt
    #newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
    if newth>(4.*np.pi):
        newth=4.*np.pi
        newthdot = 0.
    if newth<(-4.*np.pi):
        newth=-4.*np.pi
        newthdot = 0.

    return np.array([newth, newthdot])

def default_reward(sa):
    return -(1. - np.exp(-1.*(sa[0]**2)))

def default_reset():
    high = np.array([3.*np.pi, 15])
    return np.random.uniform(low=-high, high=high) + np.array([np.pi, 0]) # for generating offline data
    #return np.random.randn(2)*.5 + np.array([np.pi, 0]) # for policy evaluation
class local__envfn():
    def __init__(self):
        self.one_step = delault_one_step
        self.reward = default_reward
        self.reset = default_reset


class CustomPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.viewer = None

        high = np.array([4.*np.pi, self_max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self_max_torque,
            high=self_max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

        self.local_envfn = local__envfn()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        u = np.clip(u, -self_max_torque, self_max_torque)[0]
        self.last_u = u  # for rendering
        costs = -self.local_envfn.reward(np.concatenate( (self.state, np.array([u])),axis=0 ))
        self.state = self.local_envfn.one_step( self.state, np.array([u]) )
        return self.state, -costs, False, {}

    def reset(self):
        self.state = self.local_envfn.reset()
        self.last_u = None
        return self.state

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


