# 导入套接字的包
import socket

# 参数调整
HOST = '120.55.81.165'
PORT = 54585
BufferSize = 1024

# socket.AF_INET (IPV4)
# socket.SOCK_STREAM (TCP)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 连接服务端
s.connect((HOST, PORT))

while True:
    # 输入待发送数据
    print('Please input your data:')
    InputData = input('> ')

    # 发送数据到服务端
    s.sendall(InputData.encode('gb2312'))

    # 接收服务端返回数据
    ReceiveData = s.recv(BufferSize)
    print(ReceiveData.decode('gb2312'))

# 关闭 socket
# s.close()
