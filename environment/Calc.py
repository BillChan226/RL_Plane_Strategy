#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 本模块提供数学计算服务

from math import pow, sqrt, asin
import globalparam

def update_status(alli_flag, enemy_flag):
    stat = globalparam.get_value('status')
    if float(alli_flag) == -1:
        stat[0] = 0
    else:
        stat[0] = 1

    if enemy_flag:
        stat[1] = 2
    else:
        stat[1] = 1

    globalparam.set_value('status', stat)
    return stat


def calc_dis(x1, x2, y1, y2, z1, z2):
    return sqrt(pow((float(x1) - float(x2)), 2) + pow((float(y1) - float(y2)), 2) + pow((float(z1) - float(z2)), 2))

def tans_coordinate(x, y, z):
    newX = float(x) - globalparam.get_value('devX')
    newZ = float(z) - globalparam.get_value('devZ')
    # 柱坐标系r
    radius = sqrt(pow(newX, 2) + pow(newZ, 2))
    # 柱坐标系θ
    if newX > 0 and newZ > 0:                                         # 战机位于第一象限
        angle = asin(newZ / radius) * globalparam.get_value('radScale')

    elif newX < 0 and newZ > 0:                                       # 战机位于第二象限
        angle = (globalparam.get_value('pi') - asin(newZ / radius)) * globalparam.get_value('radScale')

    elif newX < 0 and newZ < 0:                                       # 战机位于第三象限
        angle = (globalparam.get_value('pi') - asin(newZ / radius)) * globalparam.get_value('radScale')

    else:                                                             # 战机位于第四象限
        angle = (2 * globalparam.get_value('pi') + asin(newZ / radius)) * globalparam.get_value('radScale')

    # 柱坐标系z
    height = float(y)

    return newX, newZ, radius, angle, height