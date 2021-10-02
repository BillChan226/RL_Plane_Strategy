# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Algorithms
from rl_algorithms.algos.ddpg.ddpg import ddpg as ddpg_pytorch
from rl_algorithms.algos.ppo.ppo import ppo as ppo_pytorch
from rl_algorithms.algos.sac.sac import sac as sac_pytorch
from rl_algorithms.algos.td3.td3 import td3 as td3_pytorch
from rl_algorithms.algos.trpo.trpo import trpo as trpo_pytorch

# Loggers
from rl_algorithms.utils.logx import Logger, EpochLogger

# Version
from rl_algorithms.version import __version__