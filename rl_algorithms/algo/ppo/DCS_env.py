import socket

class DCS_env:
    def __init__(self, host_ip='192.168.3.37', host_port=30000, size=1024):
        # 参数调整
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
        self.conn.sendall('ready'.encode('gb2312'))
        print('successfully send reset command to client')
        next_obserstr = self.conn.recv(self.BufferSize)
        initial_obs = self.obs_parser(next_obserstr)
        return initial_obs

    def step(self, a):
        #a = [1,2,3,4,5,6,7]
        action = '1' + '@' + str(a[0]) + ',' + str(a[1]) + ',' + str(a[2]) + ',' + str(a[3]) + ',' + str(a[3]) + ',' + str(a[4]) + ',' + str(a[5]) + ',' + str(a[6])
        #print(action)
        # 把客服端发送过来的数据又转发回去
        self.conn.sendall(action.encode('gb2312'))
        print('successfully send action to client')
        next_obserstr = self.conn.recv(self.BufferSize)
        print("Received from %s: %s" % (self.addr, data.decode('gb2312')))
        full_next_obs, next_ourplane = self.obs_parser(next_obserstr)
        rwd = self.reward(next_obserstr)
        done = not()
        # 关闭客户端连接
        # conn.colse()
        return full_next_obs, rwd, done, 0
    def reward(self, obsstr):
        ourplane, allyplane, enemyplane1, enemyplane2 = self.obs_parser(obsstr)
        rwd = -min(np.linalg.norm(ourplane - enemyplane1), np.linalg.norm(ourplane - enemyplane2))
        return rwd

    def obs_parser(self, obsstr):
        ourplane, allyplane, enemyplane1, enemyplane2 =
        observation = ourplane
        observation = observation.extend(allyplane)
        observation = observation.extend(enemyplane1)
        observation = observation.extend(enemyplane2)
        return observation, ourplane, allyplane, enemyplane1, enemyplane2
