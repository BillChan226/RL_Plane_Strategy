import socket
import numpy as np
from gym import spaces
import copy

class DCS_env:
    def __init__(self, host_ip='192.168.3.37', host_port=30000, size=1024):
        # 参数调整
        self.state_dim = 12
        self.action_dim = 7
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.HOST = host_ip
        self.PORT = host_port
        self.BufferSize = size
        # socket.AF_INET (IPV4)
        # socket.SOCK_STREAM (TCP)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 绑定客户端IP和端口号
        s.bind((self.HOST, self.PORT))
        # 最大允许连接数量
        s.listen(3)
        # 阻塞，当出现客户端的请求完成连接，并获取数据和客户端的信息
        self.conn, self.addr = s.accept()

    def reset(self):
        self.conn.sendall('Ready'.encode('gb2312'))
        while True:
            _, ourplane, _, _, _ = self.obs_parser(self.conn.recv(self.BufferSize))
            print('ourplane', ourplane)
            if ourplane[2] > 0: break
        print('lalalalala')
        next_obserstr = self.conn.recv(self.BufferSize)
        initial_obs, _, _, _, _  = self.obs_parser(next_obserstr)
        print('successfully send reset command to client')
        return initial_obs

    def step(self, a):
        #a = [1,2,3,4,5,6,7]
        action = '1' + '@' + str(a[0]) + ',' + str(a[1]) + ',' + str(a[2]) + ',' + str(a[3]) + ',' + str(a[3]) + ',' + str(a[4]) + ',' + str(a[5]) + ',' + str(a[6])
        #print(action)
        self.conn.sendall(action.encode('gb2312'))
        print('successfully send action to client')
        next_obserstr = self.conn.recv(self.BufferSize)
        print('next_obserstr', next_obserstr)
        full_next_obs, next_ourplane, _, _, _ = self.obs_parser(next_obserstr)
        rwd = self.reward(next_obserstr)
        done = 1 if next_ourplane[2] < 0 else 0
        # 关闭客户端连接
        # conn.colse()
        return full_next_obs, rwd, done, 0

    def reward(self, obsstr):
        _, ourplane, allyplane, enemyplane1, enemyplane2 = self.obs_parser(obsstr)
        #print('duudwdwd', np.linalg.norm(ourplane - enemyplane1))
        #print('debug', min(np.linalg.norm(ourplane - enemyplane1, 2), np.linalg.norm(ourplane - enemyplane2, 2)))
        rwd = -min(np.linalg.norm(np.array(ourplane) - np.array(enemyplane1), 2), np.linalg.norm(np.array(ourplane) - np.array(enemyplane2), 2))
        return rwd

    def obs_parser(self, obsstr):
        #obsstr_sib = copy.deepcopy(obsstr)
        obsstr = obsstr.decode('gb2312')
        #print("obsstr", obsstr)
        ourplane = [float(str(obsstr).split('@', 4)[0].split(':', 3)[0]), float(str(obsstr).split('@', 4)[0].split(':', 3)[1]), float(str(obsstr).split('@', 4)[0].split(':', 3)[2])]
        allyplane = [float(str(obsstr).split('@', 4)[1].split(':', 3)[0]), float(str(obsstr).split('@', 4)[1].split(':', 3)[1]), float(str(obsstr).split('@', 4)[1].split(':', 3)[2])]
        enemyplane1 = [float(str(obsstr).split('@', 4)[1].split(':', 3)[0]), float(str(obsstr).split('@', 4)[1].split(':', 3)[1]), float(str(obsstr).split('@', 4)[1].split(':', 3)[2])]
        enemyplane2 = [float(str(obsstr).split('@', 4)[2].split(':', 3)[0]), float(str(obsstr).split('@', 4)[2].split(':', 3)[1]), float(str(obsstr).split('@', 4)[2].split(':', 3)[2])]
        observation = ourplane.copy()
        observation.extend(allyplane)
        observation.extend(enemyplane1)
        observation.extend(enemyplane2)
        #print('observation', observation)
        return observation, ourplane, allyplane, enemyplane1, enemyplane2
