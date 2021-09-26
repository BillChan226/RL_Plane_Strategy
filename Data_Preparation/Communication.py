#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 本模块提供初始化和通信服务

from socket import *
from parseStr import ParseData
import globalparam
from initialize import init_origin

# 初始化不变的全局变量
globalparam._init()
globalparam.set_value('pi', 3.1415926)                                          # Π的值
globalparam.set_value('isDebug', 0)                                             # 是否在调试
globalparam.set_value('isPlaneOne', 1)                                          # 己方战机是否为一号战机
globalparam.set_value('hostDataNum', 15)                                        # 己方战机数据总数
globalparam.set_value('alliDataNum', 7)                                         # 友方战机数据总数
globalparam.set_value('enemyDataNum', 7)                                        # 敌方战机数据总数
globalparam.set_value('disScale', 1.852)                                        # 一海里等于1.852公里
globalparam.set_value('radScale', 180 / globalparam.get_value('pi'))            # 弧度制到角度制的换算比例
globalparam.set_value('battle_radius', 10)                                      # 战场区域半径(海里)

# 初始化可变的全局变量
globalparam.set_value('status', [1, 2])                                         # 初始友机数为1，敌机数为0
globalparam.set_value('devX', -277007.015206915)                                # 默认的X轴偏移
globalparam.set_value('devZ', 609429.843537115)                                 # 默认的Z轴偏移

# 建立通信

# 战机数据接收端口设置
if globalparam.get_value('isPlaneOne'):
    portSelf = 21824        # 一号战机己方通信端口
    portAlli = 21826        # 一号战机友方通信端口
    portEnemy = 21825       # 一号战机敌方通信端口
    portCtrl = 21827        # 一号战机发送指令端口

else:
    portSelf = 21924        # 二号战机己方通信端口
    portAlli = 21926        # 二号战机友方通信端口
    portEnemy = 21925       # 二号战机敌方通信端口
    portCtrl = 21827        # 二号战机发送指令端口

bufsiz = 1024               # 接收数据缓冲区大小
hostip = '10.163.196.33'     # 本机ip地址

# 设置UDP通信接收地址
addrSelf = ('', portSelf)       # 己方战机UDP通信接收地址
addrAlli = ('', portAlli)       # 友方战机UDP通信接收地址
addrEnemy = ('', portEnemy)     # 敌方战机UDP通信接收地址
addrCtrl = (hostip, portCtrl)   # 己方战机UDP通信发送地址

# 建立通信并绑定套接字
udpSelf = socket(AF_INET, SOCK_DGRAM)   # 创建己方战机数据udp端口套接字
udpSelf.bind(addrSelf)                  # 套接字与地址绑定

udpAlli = socket(AF_INET, SOCK_DGRAM)   # 创建友方战机数据udp端口套接字
udpAlli.bind(addrAlli)                  # 套接字与地址绑定

udpEnemy = socket(AF_INET, SOCK_DGRAM)  # 创建敌方战机数据udp端口套接字
udpEnemy.bind(addrEnemy)                # 套接字与地址绑定

udpCtrl = socket(AF_INET, SOCK_DGRAM)   # 创建控制数据udp端口套接字

init_origin(udpSelf, udpAlli, udpEnemy, bufsiz)     # 初始化坐标系，将坐标系原点转化为战场中心

parseEngine = ParseData()

# 接收UDP端口数据
while True:
    if not globalparam.get_value('isDebug'):
        dataSelf, addr = udpSelf.recvfrom(bufsiz)       # 接收己方战机数据，格式为bytes

        dataAlli, addr = udpAlli.recvfrom(bufsiz)       # 接收友方战机数据，格式为bytes

        dataEnemy, addr = udpEnemy.recvfrom(bufsiz)     # 接收敌机数据，格式为bytes

        # 将bytes转为str
        dataSelf = dataSelf.decode()
        dataAlli = dataAlli.decode()
        dataEnemy = dataEnemy.decode()

        # print('己方战机数据为%s' % dataSelf)
        # print('友方战机数据为%s' % dataAlli)
        # print('敌方战机数据为%s' % dataEnemy)

        parseEngine.get_flight_data(dataSelf, dataAlli, dataEnemy)

        udpCtrl.sendto('1@0,0,0.5,0,0,0,0'.encode(), addrCtrl)






    else:
        dataSelf = '0.1@16778240:-288372.64540332:1999.2688590855:604732.6754663:6.195529319346:0.068854615092278:8.7469419440822e-07:179.19110107422:-0.057070199400187:-15.747104644775:1.2650721146201e-05:-1.9459616851236e-06:-0.007682628929615:3.9632575511932:250'
        dataAlli = '0.1@16777472:-287744.4024806:1997.5829753475:616239.9178848:6.1776095479727:0.0048038540408015:0.00071518606273457&'
        dataEnemy = '0.1@16777984:-265591.75465059:1997.5829753524:613895.88626021:3.0364842414856:0.0048039588145912:0.0020442630629987&16777728:-266319.25829315:1997.6807466468:602850.89453715:3.0569741725922:0.0059103975072503:0.0020443215034902&'

        parseEngine.get_flight_data(dataSelf, dataAlli, dataEnemy)

        break

