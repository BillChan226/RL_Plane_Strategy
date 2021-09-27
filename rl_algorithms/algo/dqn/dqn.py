from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.dqn.core as core
from spinup.utils.logx import EpochLogger
import random
from datetime import datetime
from tensorboardX import SummaryWriter

from ipdb import set_trace as tt
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def Variable(var):
    return var.to(device)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DQN agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        # available only for discrete action
        self.act_buf = np.zeros(size, dtype=np.int)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        ans = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
        ans['act'] = torch.as_tensor(ans['act'], dtype=torch.long)
        return ans

def dqn(env_fn,  q_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e4), gamma=0.99,
        polyak=0.995, q_lr=1e-3, batch_size=128, start_steps=0,
        update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10,
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1, tf_logger='logs/dqn/'):

    tf_logger = tf_logger[:-1] + datetime.now().strftime('%Y%m%d%H%M%S') + '/'
    # tt()
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    # set up tensorboard parameters:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    # only be applicable for discrete action
    act_dim = env.action_space.n
    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # q_func_hidden_size = (256, 128, 64)
    q_func = core.MLPQFunction(obs_dim, act_dim, **q_kwargs).to(device)
    with SummaryWriter(log_dir=tf_logger, comment="DQN_graph") as w:
        dummy_input =torch.rand((1, obs_dim),dtype=torch.float32).to(device)
        w.add_graph(q_func, dummy_input)
    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(m) for m in [q_func])
    q_target = deepcopy(q_func)
    q_target.eval()
    logger.log('\nNumber of parameters:  q: %d\n' % var_counts)
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in q_target.parameters():
        p.requires_grad = False
    # #test q_func
    # qv = q_func(torch.randn([128, 4]))
    # action = torch.randint(high=1, low=0, size = [128, 2])
    # Set up function for computing double Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        r = Variable(r)
        o2 = Variable(o2)
        a = Variable(a)
        o = Variable(o)
        d = Variable(d)
        q = q_func(o).gather(1, a.unsqueeze(1))
        # Bellman backup for Q function
        with torch.no_grad():
            q_targ = q_target(o2).detach().max(1)[0]
            backup = r + gamma * (1 - d) * q_targ
        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.cpu().detach().numpy())
        return loss_q, loss_info
    # TODO: change learning rate
    q_optimizer = Adam(q_func.parameters())
    # q_optimizer = RMSprop(q_func.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
    # Set up model saving
    logger.setup_pytorch_saver(q_func)

    def update(data):
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        for param in q_func.parameters():
            param.grad.data.clamp_(-1, 1)

        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(q_func.parameters(), q_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
        return loss_q.item()
    # TODO: the initial value for exploration?
    # exploration = core.LinearScheduler(steps_per_epoch * epochs, init_value=0.9, final_value=0.05)
    exploration = core.ExpScheduler(init_value=0.9, final_value=0.05, decay=200)
    def get_action(obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        writer.add_scalar(tag="epsilon", scalar_value=eps_threshold, global_step=t)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # return policy_net(state).max(1)[1].view(1, 1)
                obs = torch.from_numpy(obs).unsqueeze(0).type(torch.float32)
                # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
                action = q_func(Variable(obs)).data.max(1)[1].item()
                return action
        else:
            return env.action_space.sample()

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                obs = torch.from_numpy(o).unsqueeze(0).type(torch.float32)
                action = q_func(Variable(obs)).data.max(1)[1].cpu().numpy().squeeze()
                o, r, d, _ = test_env.step(action)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    writer = SummaryWriter(logdir=tf_logger)
    # Main loop: collect experience in env and update/log each epoch

    epoch_counter = 0
    for t in range(total_steps):
        if t > start_steps:
            a = get_action(o, (t + 1) // steps_per_epoch)
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d
        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)
        o = o2

        if d or ep_len == max_ep_len:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            epoch_counter += 1
            writer.add_scalar('train_reward', ep_ret, global_step=epoch_counter)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        loss_q = 0
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                loss_q += update(data=batch)

            loss_q /= update_every
            # core.tensorboard_logger(logdir=tf_logger, scalar=loss_q, step=t, tag='q_loss')
            writer.add_scalar('q_loss', loss_q, global_step=t)
        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            # core.tensorboard_logger(logdir=tf_logger, scalar=ep_ret, step=epoch, tag='train_reward')
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)
            test_agent()
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()
    writer.close()
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    # TODO: the original default is 256
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='dqn')
    parser.add_argument('--logger', type=str, default='logs/dqn/')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    dqn(lambda: gym.make(args.env),
         q_kwargs=dict(hidden_sizes=[args.hid] * args.l),
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs,
        tf_logger=args.logger)
