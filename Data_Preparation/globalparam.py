#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 本模块提供全局变量管理服务

def _init():
    global globalDict
    globalDict = {}

def set_value(name, value):
    globalDict[name] = value

def get_value(name, defValue = None):
    try:
        return globalDict[name]
    except KeyError:
        print("%s not found" % name)
        return defValue