#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 本模块提供解析数据服务

import Calc
import globalparam

class ParseData():
    def __init__(self):
        self.hostDataList = []
        # hostDataList[0]:  本机ID
        # hostDataList[1]:  本机x轴位置坐标(m)
        # hostDataList[2]:  本机y轴位置坐标(m)
        # hostDataList[3]:  本机z轴位置坐标(m)
        # hostDataList[4]:  本机航向角(rad)
        # hostDataList[5]:  本机俯仰角(rad)
        # hostDataList[6]:  本机滚转角(rad)
        # hostDataList[7]:  本机x轴速度(m/s)
        # hostDataList[8]:  本机y轴速度(m/s)
        # hostDataList[9]:  本机z轴速度(m/s)
        # hostDataList[10]: 本机x轴角速度(rad/s)
        # hostDataList[11]: 本机y轴角速度(rad/s)
        # hostDataList[12]: 本机z轴角速度(rad/s)
        # hostDataList[13]: 本机攻角(rad/s)
        # hostDataList[14]: 本机航炮数量
        self.alliDataList = []
        # alliDataList[0]:  友机ID
        # alliDataList[1]:  友机x轴位置坐标(m)
        # alliDataList[2]:  友机y轴位置坐标(m)
        # alliDataList[3]:  友机z轴位置坐标(m)
        # alliDataList[4]:  友机航向角(rad)
        # alliDataList[5]:  友机俯仰角(rad)
        # alliDataList[6]:  友机滚转角(rad)
        self.enemy1DataList = []
        # enemy1DataList[0]:  一号敌机ID
        # enemy1DataList[1]:  一号敌机x轴位置坐标(m)
        # enemy1DataList[2]:  一号敌机y轴位置坐标(m)
        # enemy1DataList[3]:  一号敌机z轴位置坐标(m)
        # enemy1DataList[4]:  一号敌机航向角(rad)
        # enemy1DataList[5]:  一号敌机俯仰角(rad)
        # enemy1DataList[6]:  一号敌机滚转角(rad)
        self.enemy2DataList = []
        # enemy2DataList[0]:  二号敌机ID
        # enemy2DataList[1]:  二号敌机x轴位置坐标(m)
        # enemy2DataList[2]:  二号敌机y轴位置坐标(m)
        # enemy2DataList[3]:  二号敌机z轴位置坐标(m)
        # enemy2DataList[4]:  二号敌机航向角(rad)
        # enemy2DataList[5]:  二号敌机俯仰角(rad)
        # enemy2DataList[6]:  二号敌机滚转角(rad)

    def get_flight_data(self, selfdata, allidata, enemydata):
        # 更新其它战机飞行状态
        self.status = Calc.update_status(allidata.split('@', 2)[0], enemydata.split('@', 2)[1].split('&', 3)[1])
        # print(self.status)

        # 创建存活飞行的数据列表
        self.hostDataList = selfdata.split('@', 2)[1].split(':', globalparam.get_value('hostDataNum'))
        self.alliDataList = allidata.split('@', 2)[1].split('&', 2)[0].split(':', globalparam.get_value('alliDataNum')) \
        if self.status[0] == 1 else [0, 0, 0, 0, 0, 0, 0]
        self.enemy1DataList = enemydata.split('@', 2)[1].split('&', 3)[0].split(':', globalparam.get_value('enemyDataNum'))
        self.enemy2DataList = enemydata.split('@', 2)[1].split('&', 3)[1].split(':', globalparam.get_value('enemyDataNum'))\
        if self.status[1] == 2 else [0, 0, 0, 0, 0, 0, 0]

        # 计算与友机的距离
        self.disAlli = Calc.calc_dis(self.hostDataList[1], self.alliDataList[1], self.hostDataList[2], \
                                         self.alliDataList[2], self.hostDataList[3], self.alliDataList[3]) \
                                        if self.status[0] == 1 else 0

        # 计算与一号敌机的距离
        self.disEnemy1 = Calc.calc_dis(self.hostDataList[1], self.enemy1DataList[1], self.hostDataList[2], \
                                           self.enemy1DataList[2], self.hostDataList[3], self.enemy1DataList[3])

        # 计算与二号敌机的距离
        self.disEnemy2 = Calc.calc_dis(self.hostDataList[1], self.enemy2DataList[1], self.hostDataList[2], \
                                           self.enemy2DataList[2], self.hostDataList[3], self.enemy2DataList[3]) \
                                            if self.status[0] == 1 else 0

        # print("剩余航炮数为%d\n" % int(self.hostDataList[14]))

        if globalparam.get_value('isDebug'):
            print(self.disAlli, self.disEnemy1, self.disEnemy2)

        # 将各战机的原始坐标系转换为战场中心为轴的柱坐标系
        self.x_s, self.z_s, self.radius_s, self.angle_s, self.height_s = Calc.tans_coordinate(self.hostDataList[1], self.hostDataList[2], self.hostDataList[3])

        if self.status[0] == 1:
            self.x_a, self.z_a, self.radius_a, self.angle_a, self.height_a = Calc.tans_coordinate(self.alliDataList[1], self.alliDataList[2], self.alliDataList[3])
        else:
            self.x_a = 0
            self.z_a = 0
            self.radius_a = -1
            self.angle_a = -1
            self.height_a = -1

        self.x_e1, self.z_e1, self.radius_e1, self.angle_e1, self.height_e1 = Calc.tans_coordinate(self.enemy1DataList[1], self.enemy1DataList[2], self.enemy1DataList[3])

        if self.status[1] == 2:
            self.x_e2, self.z_e2, self.radius_e2, self.angle_e2, self.height_e2 = Calc.tans_coordinate(self.enemy2DataList[1], self.enemy2DataList[2],
                                                                  self.enemy2DataList[3])
        else:
            self.x_e2 = 0
            self.z_e2 = 0
            self.radius_e2 = -1
            self.angle_e2 = -1
            self.height_e2 = -1

        globalparam.set_value('r', self.radius_e2)

        # print('我方战机的位置为%f, %f, %f' % (radius_s, angle_s, height_s))
        # print('友方战机的位置为%f, %f, %f' % (radius_a, angle_a, height_a))
        # print('一号敌机的位置为%f, %f, %f' % (radius_e1, angle_e1, height_e1))
        # print('二号敌机的位置为%f, %f, %f' % (radius_e2, angle_e2, height_e2))


        # elif self.status == [0, 2]:
        #     # 创建存活飞行的数据列表
        #     self.hostDataList = selfdata.split('@', 2)[1].split(':', globalparam.get_value('hostDataNum'))
        #     self.enemy1DataList = enemydata.split('@', 2)[1].split('&', 3)[0].split(':', globalparam.get_value('enemyDataNum'))
        #     self.enemy2DataList = enemydata.split('@', 2)[1].split('&', 3)[1].split(':', globalparam.get_value('enemyDataNum'))
        #
        #     # 计算与一号敌机的距离
        #     self.disEnemy1 = Calc.calc_dis(self.hostDataList[1], self.enemy1DataList[1], self.hostDataList[2], \
        #                                    self.enemy1DataList[2], self.hostDataList[3], self.enemy1DataList[3])
        #
        #     # 计算与二号敌机的距离
        #     self.disEnemy2 = Calc.calc_dis(self.hostDataList[1], self.enemy2DataList[1], self.hostDataList[2], \
        #                                    self.enemy2DataList[2], self.hostDataList[3], self.enemy2DataList[3])
        #     if globalparam.get_value('isDebug'):
        #         print(self.disEnemy1, self.disEnemy2)
        #
        #     # 将各战机的原始坐标系转换为战场中心为轴的柱坐标系
        #     radius_s, angle_s, height_s = Calc.tans_coordinate(self.hostDataList[1], self.hostDataList[2], self.hostDataList[3])
        #     radius_e1, angle_e1, height_e1 = Calc.tans_coordinate(self.enemy1DataList[1], self.enemy1DataList[2], self.enemy1DataList[3])
        #     radius_e2, angle_e2, height_e2 = Calc.tans_coordinate(self.enemy2DataList[1], self.enemy2DataList[2], self.enemy2DataList[3])
        #
        #     # print('我方战机的位置为%f, %f, %f' % (radius_s, angle_s, height_s))
        #     # print('一号敌机的位置为%f, %f, %f' % (radius_e1, angle_e1, height_e1))
        #     # print('二号敌机的位置为%f, %f, %f' % (radius_e2, angle_e2, height_e2))
        #
        #
        # elif self.status == [1, 1]:
        #     # 创建存活飞行的数据列表
        #     self.hostDataList = selfdata.split('@', 2)[1].split(':', globalparam.get_value('hostDataNum'))
        #     self.alliDataList = allidata.split('@', 2)[1].split('&', 2)[0].split(':', globalparam.get_value('alliDataNum'))
        #     self.enemy1DataList = enemydata.split('@', 2)[1].split('&', 3)[0].split(':', globalparam.get_value('enemyDataNum'))
        #
        #
        #     # 计算与友机的距离
        #     self.disAlli = Calc.calc_dis(self.hostDataList[1], self.alliDataList[1], self.hostDataList[2], \
        #                                  self.alliDataList[2], self.hostDataList[3], self.alliDataList[3])
        #
        #     # 计算与一号敌机的距离
        #     self.disEnemy1 = Calc.calc_dis(self.hostDataList[1], self.enemy1DataList[1], self.hostDataList[2], \
        #                                    self.enemy1DataList[2], self.hostDataList[3], self.enemy1DataList[3])
        #
        #     if globalparam.get_value('isDebug'):
        #         print(self.disAlli, self.disEnemy1)
        #
        #     # 将各战机的原始坐标系转换为战场中心为轴的柱坐标系
        #     radius_s, angle_s, height_s = Calc.tans_coordinate(self.hostDataList[1], self.hostDataList[2], self.hostDataList[3])
        #     radius_a, angle_a, height_a = Calc.tans_coordinate(self.alliDataList[1], self.alliDataList[2], self.alliDataList[3])
        #     radius_e1, angle_e1, height_e1 = Calc.tans_coordinate(self.enemy1DataList[1], self.enemy1DataList[2], self.enemy1DataList[3])
        #
        #     # print('我方战机的位置为%f, %f, %f' % (radius_s, angle_s, height_s))
        #     # print('友方战机的位置为%f, %f, %f' % (radius_a, angle_a, height_a))
        #     # print('一号敌机的位置为%f, %f, %f' % (radius_e1, angle_e1, height_e1))
        #
        #
        # else:
        #     # 创建存活飞行的数据列表
        #     self.hostDataList = selfdata.split('@', 2)[1].split(':', globalparam.get_value('hostDataNum'))
        #     self.enemy1DataList = enemydata.split('@', 2)[1].split('&', 3)[0].split(':', globalparam.get_value('enemyDataNum'))
        #
        #     # 计算与一号敌机的距离
        #     self.disEnemy1 = Calc.calc_dis(self.hostDataList[1], self.enemy1DataList[1], self.hostDataList[2], \
        #                                    self.enemy1DataList[2], self.hostDataList[3], self.enemy1DataList[3])
        #
        #     if globalparam.get_value('isDebug'):
        #         print(self.disEnemy1)
        #
        #     # 将各战机的原始坐标系转换为战场中心为轴的柱坐标系
        #     radius_s, angle_s, height_s = Calc.tans_coordinate(self.hostDataList[1], self.hostDataList[2], self.hostDataList[3])
        #     radius_e1, angle_e1, height_e1 = Calc.tans_coordinate(self.enemy1DataList[1], self.enemy1DataList[2], self.enemy1DataList[3])
        #
        #     # print('我方战机的位置为%f, %f, %f' % (radius_s, angle_s, height_s))
        #     # print('一号敌机的位置为%f, %f, %f' % (radius_e1, angle_e1, height_e1))

    def encode_data(self):
        self.dtEncode = str(self.x_s) + ':' + str(self.z_s) + ':' + str(self.height_s) + '@' + \
                        str(self.x_a) + ':' + str(self.z_a) + ':' + str(self.height_a) + '@' + \
                        str(self.x_e1) + ':' + str(self.z_e1) + ':' + str(self.height_e1) + '@' + \
                        str(self.x_e2) + ':' + str(self.z_e2) + ':' + str(self.height_e2)
        return self.dtEncode









