import socket
import time

# 参数调整
HOST = '192.168.3.37'
PORT = 30000
BufferSize = 1024

# socket.AF_INET (IPV4)
# socket.SOCK_STREAM (TCP)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定客户端IP和端口号
s.bind((HOST, PORT))

# 最大允许连接数量
s.listen(3)

# 阻塞，当出现客户端的请求完成连接，并获取数据和客户端的信息
conn, addr = s.accept()



# 读取客户端发送过来的数据
#data = conn.recv(BufferSize)
#print("Received from %s: %s" % (addr, data.decode('gb2312')))


while True:
    # 把客服端发送过来的数据又转发回去
    #data = conn.recv(BufferSize)
    #print("Received from %s: %s" % (addr, data.decode('gb2312')))
    #conn.sendall('I got it!'.encode('gb2312'))
    #time.sleep(5)
    #a = [1,2,3,4,5,6,7]
    action = '1' + '@' + str(0.5) + ',' + str(0.5) + ',' + str(0.5) + ',' + str(0.5) + ',' + str(0.5) + ',' + str(0.5) + ',' + str(0.5) + ',' + str(0.5)
    #print(action)
    conn.sendall(action.encode('gb2312'))
    print('successfully send action to client')
    next_obserstr = conn.recv(BufferSize)
    #next_obserstr = data.decode('gb2312')
    #print("Received from %s: %s" % (self.addr, data.decode('gb2312')))
    print('next_obserstr', next_obserstr)
    #full_next_obs, next_ourplane, _, _, _ = self.obs_parser(next_obserstr)
    #rwd = self.reward(next_obserstr)
    #done = 1 if next_ourplane[2] < 0 else 0
    # 关闭客户端连接
    # conn.colse()

    # 关闭客户端连接
    # conn.colse()

