import gym
from gym.utils import seeding
import numpy as np
import pickle
import torch
from spinup.utils.test_policy import Demo_eps_buffer
from scipy.stats import norm

class DemoGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, demo_file, seed=0):
        self.seed(seed)
        self.demo_file = demo_file
        self.epoch_data = None
        self.step_count = 0
        self.obs_buf, self.obs2_buf, self.act_buf, self.rew_buf, self.done_buf = None, None, None, None, None
        self.buffer_keys = ['obs_buf', 'obs2_buf', 'act_buf', 'rew_buf', 'done_buf']
        self.load_replay()

    def free_step(self):
        # return the data from buffer, including the action it takes
        self.step_count += 1
        assert self.step_count < len(self.obs_buf)
        obs = self.obs_buf[self.step_count]
        buf_a = self.act_buf[self.step_count]
        buf_r = self.rew_buf[self.step_count]
        done = self.done_buf[self.step_count]
        return np.array(obs), buf_r, done, {}, buf_a

    def load_replay(self):
        buffer = Demo_eps_buffer()
        buffer.load(self.demo_file)
        self.epoch_data = buffer.episode_buf
        return

    def step(self, action, sigma=None):
        """
        Here only for the continue version
        """
        self.step_count += 1
        assert self.step_count < len(self.obs_buf)
        obs = self.obs_buf[self.step_count]
        obs2 = self.obs2_buf[self.step_count]
        buf_act = self.act_buf[self.step_count]
        buf_r = self.rew_buf[self.step_count]
        done = self.done_buf[self.step_count]
        r = self.reward(action, buf_act, sigma, buf_r=buf_r)
        return np.array(obs), r, done, {}

    def reward(self, act, buf_act, stddev=1, buf_r=0.0):
        # only for continual control
        d = 0.5 - abs(0.5-norm.cdf(x=(act-buf_act)/stddev))
        log_p =  sum(- (act - buf_act)**2 / (2*stddev**2) - 0.5 * np.log(2*np.pi*stddev**2))
        r = np.exp(log_p) # * abs(buf_r)

        return r

    def reset(self):
        epochs = list(self.epoch_data.keys())
        # randomly select an episode of trajectory
        eps = np.random.randint(low=1, high=len(epochs)+1)
        current_episode = self.epoch_data[eps]
        self.obs_buf = current_episode['obs_buf']
        self.obs2_buf = current_episode['obs2_buf']
        self.act_buf = current_episode['act_buf']
        self.rew_buf = current_episode['rew_buf']
        self.done_buf = current_episode['done_buf']
        self.step_count = 0
        return self.obs_buf[self.step_count]

    def seed(self, seed=None):
        """
        Seed random generator
        :param seed: (int)
        :return: ([int])
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def check_env(self, gym_env):
        obs_dim = gym_env.observation_space.shape
        act_dim = gym_env.action_space.shape
        # the first observation
        demo_obs = self.epoch_data[1]['obs_buf'][1].shape
        demo_act_dim = self.epoch_data[1]['act_buf'][1].shape
        assert obs_dim == demo_obs and demo_act_dim == act_dim, "Error: please use the same env for pre-training"
        return True


if __name__ == "__main__":
    k = lambda: gym.make(DemoGymEnv)
    env = DemoGymEnv(demo_file='data/Ant50epoch.pickle')
    a = env.reset()
    b= env.step(1)
    b= env.step(2)
