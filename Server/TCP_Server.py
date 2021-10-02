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

counter = 0
while True:
    # 把客服端发送过来的数据又转发回去
    data = conn.recv(BufferSize)
    print("Received from %s: %s" % (addr, data.decode('gb2312')))
    counter += 1
    conn.sendall('I got it! {}'.format(counter).encode('gb2312'))
    time.sleep(5)

    # 关闭客户端连接
    # conn.colse()

