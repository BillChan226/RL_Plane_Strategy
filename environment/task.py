#!/usr/bin/python
# -*- coding: UTF-8 -*-

from socket import *
import paramiko


# def com_with_server(serverData, ReceiveData, StatusData):
# serverIP = '120.55.81.165'
#         serverPort = 54585
#     bufsiz = 1024
#
#     # socket.AF_INET (IPV4)
#     # socket.SOCK_STREAM (TCP)
#     trainingServer = socket(AF_INET, SOCK_STREAM)
#
#     # 连接服务端
#     trainingServer.connect((serverIP, serverPort))
#
#     lastData = '123'
#
#     while True:
#         if lastData != serverData[0]:
#         # 发送数据到服务端
#             trainingServer.sendall(serverData[0].encode())
#
#             data = trainingServer.recv(bufsiz)
#
#
#         lastData = serverData[0]
#
#         # 接收服务端返回数据
#
#         if str(data.decode()) == 'Ready':
#             StatusData[0] = 'Ready'
#
#         else:
#             ReceiveData[0] = data.decode()



# def com_with_server(serverData, ReceiveData):
#     serverIP = '10.135.22.73'
#     serverPort = 28888
#     bufsiz = 1024
#
#     # socket.AF_INET (IPV4)
#     # socket.SOCK_STREAM (TCP)
#     trainingServer = socket(AF_INET, SOCK_DGRAM)
#
#     while True:
#         # 发送数据到服务端
#         trainingServer.sendto(serverData[0].encode(), (serverIP, serverPort))
#
#         # 接收服务端返回数据
#         data, addr = trainingServer.recvfrom(bufsiz)
#
#         ReceiveData[0] = data.decode()
#
#         print(ReceiveData[0])



# def com_with_server(serverData, ReceiveData):
#     serverIP = '10.135.22.73'
#     serverPort = 28888
#     bufsiz = 1024
#
#     # socket.AF_INET (IPV4)
#     # socket.SOCK_STREAM (TCP)
#     trainingServer = socket(AF_INET, SOCK_DGRAM)
#
#     while True:
#         # 发送数据到服务端
#         trainingServer.sendto(serverData[0].encode(), (serverIP, serverPort))
#
#         # 接收服务端返回数据
#         data, addr = trainingServer.recvfrom(bufsiz)
#
#         ReceiveData[0] = data.decode()
#
#         print(ReceiveData[0])



def com_with_server(serverData, ReceiveData):
    serverIP = '120.55.81.165'
    serverPort = 51797
    # TCPPort = 54585



    localpathOb = 'D:\\PlaneFile\\observation.txt'
    remotepathOb = '//home//tete//work//RLP2021//server//transfile//observation.txt'

    localpathAc = 'D:\\PlaneFile\\action.txt'
    remotepathAc = '//home//tete//work//RLP2021//server//transfile//action.txt'

    # trans = paramiko.Transport(serverIP, serverPort)
    # print('5')
    # trans.connect(username = 'tete', password = 'inmoov213')
    # print('6')
    # sftp = paramiko.SFTPClient.from_transport(trans)
    # print('7')
    trans = paramiko.SSHClient()
    trans.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    trans.connect(serverIP, serverPort, 'tete', 'INmoov213!')
    lastData = '123'

    while True:
        ftp = trans.open_sftp()

        ftp.get(remotepathAc, localpathAc)

        file = open(localpathAc, 'rb+')

        data = file.read()


        # data = trainingServer.recv(1024)
        # print('服务器发回的数据是%s' % data.decode())

        # if str(data.decode()) == 'Ready':
        #     StatusData[0] = 'Ready'
        # else:
        # print(data.decode())
        ReceiveData[0] = data.decode()

        if lastData != serverData[0]:
            # print(lastData)

            file = open(localpathOb, 'wb+')

            # print('本地数据为%s' % serverData[0])

            file.write(serverData[0].encode())

            file.close()

            ftp.put(localpathOb, remotepathOb)

            lastData = serverData[0]