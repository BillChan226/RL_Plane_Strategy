import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tensorboardX import SummaryWriter
from ipdb import set_trace as tt

class ExpScheduler:
    def __init__(self, init_value, final_value, decay):
        self.init_value = init_value
        self.final_value = final_value
        self.decay = decay

    def value(self, step):
        eps = self.final_value + (self.init_value - self.final_value) * np.exp(-1. * step / self.decay)
        return eps

class LinearScheduler:
    def __init__(self, total_step, init_value, final_value):
        self.total_step = total_step
        self.init_value = init_value
        self.final_value = final_value

    def value(self, t):
        v = t * 1.0 /self.total_step * (self.final_value - self.init_value) + self.init_value
        ans = min(1.0, v)
        return ans

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):

        q = self.q(obs)

        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class CNNQFunction(nn.Module):
    # copied from the official pytorch tutorial
    def __init__(self, h, w, outputs):
        super(CNNQFunction, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def tensorboard_logger(logdir, scalar, step, comment=None, tag='scalar'):
    with SummaryWriter(log_dir=logdir, comment=comment) as w:
        if tag == 'scalar':
            w.add_scalar(tag=tag,scalar_value=scalar, global_step=step)