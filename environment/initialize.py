#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 本模块提供初始化坐标转换服务

import Calc
import globalparam
from math import fabs

def init_origin(udpSelf, udpAlli, udpEnemy, bufsiz):
    if not globalparam.get_value('isDebug'):
        dataSelf, addr = udpSelf.recvfrom(bufsiz)       # 接收己方战机数据，格式为bytes
        dataAlli, addr = udpAlli.recvfrom(bufsiz)       # 接收友方战机数据，格式为bytes
        dataEnemy, addr = udpEnemy.recvfrom(bufsiz)     # 接收敌机数据，格式为bytes

        dataSelf = dataSelf.decode()
        dataAlli = dataAlli.decode()
        dataEnemy = dataEnemy.decode()

    else:
        dataSelf = '0.1@16778240:-288372.64540332:1999.2688590855:604732.6754663:6.195529319346:0.068854615092278:8.7469419440822e-07:179.19110107422:-0.057070199400187:-15.747104644775:1.2650721146201e-05:-1.9459616851236e-06:-0.007682628929615:3.9632575511932:250'
        dataAlli = '0.1@16777472:-287744.4024806:1997.5829753475:616239.9178848:6.1776095479727:0.0048038540408015:0.00071518606273457&'
        dataEnemy = '0.1@16777984:-265591.75465059:1997.5829753524:613895.88626021:3.0364842414856:0.0048039588145912:0.0020442630629987&16777728:-266319.25829315:1997.6807466468:602850.89453715:3.0569741725922:0.0059103975072503:0.0020443215034902&'



    x1 = float(dataSelf.split('@', 2)[1].split(':', globalparam.get_value('hostDataNum'))[1])                       # 己方战机初始X坐标
    z1 = float(dataSelf.split('@', 2)[1].split(':', globalparam.get_value('hostDataNum'))[3])                       # 己方战机初始Z坐标
    x2 = float(dataAlli.split('@', 2)[1].split('&', 2)[0].split(':', globalparam.get_value('alliDataNum'))[1])      # 友方战机初始X坐标
    z2 = float(dataAlli.split('@', 2)[1].split('&', 2)[0].split(':', globalparam.get_value('alliDataNum'))[3])      # 友方战机初始Z坐标
    x3 = float(dataEnemy.split('@', 2)[1].split('&', 3)[0].split(':', globalparam.get_value('enemyDataNum'))[1])    # 一号敌机初始X坐标
    z3 = float(dataEnemy.split('@', 2)[1].split('&', 3)[0].split(':', globalparam.get_value('enemyDataNum'))[3])    # 一号敌机初始Z坐标
    x4 = float(dataEnemy.split('@', 2)[1].split('&', 3)[1].split(':', globalparam.get_value('enemyDataNum'))[1])    # 二号敌机初始X坐标
    z4 = float(dataEnemy.split('@', 2)[1].split('&', 3)[1].split(':', globalparam.get_value('enemyDataNum'))[3])    # 二号敌机初始Z坐标

    # 不考虑高度，作战区域为圆形，四架战机初始位置为四个顶点构成四边形，将四边形两条对角线上的战机的坐标计算平均值求得战场中心的(x,z)坐标
    if fabs(z1 - z3) > fabs(z1 - z4):
        x_1 = (x1 + x3) / 2
        z_1 = (z1 + z3) / 2
        x_2 = (x2 + x4) / 2
        z_2 = (z2 + z4) / 2
    else:
        x_1 = (x1 + x4) / 2
        z_1 = (z1 + z4) / 2
        x_2 = (x2 + x3) / 2
        z_2 = (z2 + z3) / 2

    # 计算作战区域中心并将其设置为全局变量
    devX = (x_1 + x_2) / 2
    devZ = (z_1 + z_2) / 2
    globalparam.set_value('devX', float(devX))
    globalparam.set_value('devZ', float(devZ))

    # print(devX, devZ)