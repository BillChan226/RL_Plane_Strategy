from socket import *
serverPort = 28888
trainingServer = socket(AF_INET, SOCK_STREAM)
# 绑定客户端IP和端口号
trainingServer.bind(('10.163.174.207', serverPort))
# 最大允许连接数量
trainingServer.listen(3)
# 阻塞，当出现客户端的请求完成连接，并获取数据和客户端的信息
conn, addr = trainingServer.accept()
conn.sendall('Ready'.encode())
n = 0
while True:
    data = conn.recv(1024)
    print(data.decode())
    conn.sendall(str(n).encode())
    n+=1