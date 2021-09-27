import time
import joblib
import os
import os.path as osp
import pickle
import tensorflow as tf
import torch
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
from ipdb import set_trace as tt
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
def Varaible(var):
    return var.to(device)

def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic)
    else:
        get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    return get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname).to(device)
    model.eval()

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, device=device, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)
        try:
            a = get_action(o).cpu().numpy()
        except:
            a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

class Demo_eps_buffer:
    def __init__(self, size=100):
        self.key = ['obs_buf', 'obs2_buf', 'act_buf', 'rew_buf', 'done_buf']
        self.size = int(size)
        self.episode_buf = {}
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        data = (obs, next_obs, act, rew, done)
        cur_buf = self.episode_buf[self.ptr]
        for i, k in enumerate(self.key):
            cur_buf[k].append(data[i])

    def reset_path(self):
        self.ptr += 1
        self.size += 1
        self.episode_buf[self.ptr] = {}
        for k in self.key:
            self.episode_buf[self.ptr][k] = []
        return
    def remove_last(self):
        self.episode_buf[self.ptr] = {}
        for k in self.key:
            self.episode_buf[self.ptr][k] = []
        return

    def save(self, output_file):
        demo_data = self.episode_buf
        with open(output_file, 'wb') as fp:
            pickle.dump(demo_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, input_file):
        with open(input_file, 'rb') as fp:
            demo_data = pickle.load(fp)
        self.size = len(demo_data.keys())
        self.episode_buf = demo_data

class Demo_eps_buffer_BV:
    def __init__(self, size=100):
        self.key = ['obs_buf', 'obs2_buf', 'act_buf', 'rew_buf', 'logp_buf', 'done_buf']
        self.size = int(size)
        self.episode_buf = {}
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        data = (obs, next_obs, act, rew, done)
        cur_buf = self.episode_buf[self.ptr]
        for i, k in enumerate(self.key):
            cur_buf[k].append(data[i])

    def reset_path(self):
        self.ptr += 1
        self.size += 1
        self.episode_buf[self.ptr] = {}
        for k in self.key:
            self.episode_buf[self.ptr][k] = []
        return
    def remove_last(self):
        self.episode_buf[self.ptr] = {}
        for k in self.key:
            self.episode_buf[self.ptr][k] = []
        return

    def save(self, output_file):
        demo_data = self.episode_buf
        with open(output_file, 'wb') as fp:
            pickle.dump(demo_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, input_file):
        with open(input_file, 'rb') as fp:
            demo_data = pickle.load(fp)
        self.size = len(demo_data.keys())
        self.episode_buf = demo_data


class Demo_buffer:
    def __init__(self, size=100):
        self.size = int(size)
        self.obs_buf = []
        self.obs2_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        if self.ptr >= self.max_size:
            self.ptr += 1
            idx = int((self.ptr) % self.max_size)
            self.obs_buf[idx] = obs
            self.obs2_buf[idx] = next_obs
            self.act_buf[idx] = act
            self.rew_buf[idx] = rew
            self.done_buf[idx] = done
        else:
            self.size += 1
            self.ptr += 1
            self.obs2_buf.append(next_obs)
            self.obs_buf.append(obs)
            self.act_buf.append(act)
            self.rew_buf.append(rew)
            self.done_buf.append(done)
    def save(self, output_file):
        key = ['obs_buf', 'obs2_buf', 'act_buf', 'rew_buf', 'done_buf']
        data = [self.obs_buf, self.obs2_buf, self.act_buf, self.rew_buf, self.done_buf]
        demo_data = {}
        for i in range(len(key)):
            demo_data[key[i]] = data[i]
        with open(output_file, 'wb') as fp:
            pickle.dump(demo_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, input_file):

        with open(input_file, 'rb') as fp:
            demo_data = pickle.load(fp)
        self.obs2_buf = demo_data['obs2_buf']
        self.obs_buf = demo_data['obs_buf']
        self.act_buf = demo_data['act_buf']
        self.rew_buf = demo_data['rew_buf']
        self.done_buf = demo_data['done_buf']


def save_demonstration_data(output_file, env, get_action, max_ep_len=None, num_episodes=100, ):
    assert env is not None, "Environment should not be None"
    logger = EpochLogger()
    demo = Demo_buffer(size=int(1e5))
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0

    while demo.ptr <= num_episodes:
        a = get_action(o).cpu().numpy()
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        # save to the demonstration buffer
        demo.store(o, a, r, o2, d)
        o = o2
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    demo.save(output_file=output_file)

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

def save_demonstration_data_epoch(output_file, env, get_action, max_ep_len=None, num_episodes=100):
    assert env is not None, "Environment should not be None"
    logger = EpochLogger()
    demo = Demo_eps_buffer(size=int(1e5))
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 1
    demo.reset_path()
    demo.store(o, None, r, None, False)
    max_reward = - float("inf")

    while n <= num_episodes:
        try:
            a = get_action(o).cpu().numpy()
        except:
            a = get_action(o)
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        # save to the demonstration buffer
        demo.store(o, a, r, o2, d)
        o = o2
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            current_ep_reward = ep_ret
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            if ep_ret >= max_reward:
                max_reward = ep_ret
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            if  current_ep_reward > 0.8 * max_reward:

                demo.reset_path()
                demo.store(o, None, r, None, False)
            else:
                print("Current max reward {}".format(max_reward))
                demo.remove_last()
                demo.store(o, None, r, None, False)
            n = demo.ptr
    for k in range(num_episodes+1, num_episodes+5):
        if k in demo.episode_buf:
            demo.episode_buf.pop(k)
    demo.save(output_file=output_file)
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=10)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--demo-file', type=str, default="data/test.pickle")
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)
    # run_policy(env, get_action, args.len, args.episodes, not(args.norender))
    save_demonstration_data_epoch(args.demo_file, env, get_action, args.len, args.episodes)
    new_buffer = Demo_eps_buffer(1)
    new_buffer.load(args.demo_file)
    # for k in new_buffer.episode_buf.keys():
    #     print(new_buffer.episode_buf[k]['act_buf'].__len__())

