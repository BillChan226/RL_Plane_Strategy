#!/usr/bin/python
# -*- coding: UTF-8 -*-
from socket import *
from time import ctime

print("==============时间戳UDP服务器=========================")

host = '10.163.238.29'  # 主机号为空白表示可以使用任何可用的地址
port = 21867  # 端口号
bufsiz = 1024  # 接受数据缓冲大小
addr = (host, port)

udpSerSock = socket(AF_INET, SOCK_DGRAM)  # 创建udp服务器套接字
udpSerSock.bind(addr)  # 套接字与地址绑定

while True:
    print('等待接收消息...')
    data, addr = udpSerSock.recvfrom(bufsiz)    # 连续接受指定字节的数据，接收到的是字节数组
    print(data, addr)
    udpSerSock.sendto(data, addr)  # 向客户端发送时间戳数据，必须发送字节数组


udpSerSock.close()  # 关闭服务器