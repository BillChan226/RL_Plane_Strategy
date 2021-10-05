# # from socket import *
# # # ip = '10.135.3.188'
# # # port = 21856
# # # udpsock = socket(AF_INET, SOCK_DGRAM)
# # #
# # # addr = (ip, port)
# # # udpsock.bind(addr)
# # # udpsock.sendto('27.15@16777984:-271534.11758912:1987.2692805314:613149.89342179:2.9759404659271:0.041451562196016:0.023613570258021&16777728:-273292.96593952:3141.5610745817:602960.75535987:3.0530219078064:-0.078517846763134:0.0045512267388403&'.encode(), addr)
# # # data, addr = udpsock.recvfrom(1024)
# # # data = data.decode()
# # # list = data.split('@', 2)[1].split('&', 2)
# # # print(list[0])
#
# import win32api
# import win32con
# import time
# #
# # count = 10
# # while True:
# #     if count >= 0:
# #         time.sleep(1)
# #         win32api.keybd_event(81,0,0,0)
# #         win32api.keybd_event(81,0,win32con.KEYEVENTF_KEYUP,0)
# #         time.sleep(0.1)
# #         win32api.keybd_event(90,0,0,0)
# #         win32api.keybd_event(90,0,win32con.KEYEVENTF_KEYUP,0)
# #         time.sleep(0.1)
# #         win32api.keybd_event(win32con.VK_SPACE,0,0,0)
# #         win32api.keybd_event(win32con.VK_SPACE,0,win32con.KEYEVENTF_KEYUP,0)
# #         time.sleep(0.1)
# #         win32api.keybd_event(68,0,0,0)
# #         win32api.keybd_event(68,0,win32con.KEYEVENTF_KEYUP,0)
# #         time.sleep(0.1)
# #         win32api.keybd_event(65,0,0,0)
# #         win32api.keybd_event(65,0,win32con.KEYEVENTF_KEYUP,0)
# #         time.sleep(0.1)
# #         win32api.keybd_event(win32con.VK_SPACE,0,0,0)
# #         win32api.keybd_event(win32con.VK_SPACE,0,win32con.KEYEVENTF_KEYUP,0)
# #         time.sleep(0.1)
# #         win32api.keybd_event(76,0,0,0)
# #         win32api.keybd_event(76,0,win32con.KEYEVENTF_KEYUP,0)
# #         time.sleep(0.1)
# #         win32api.keybd_event(65,0,0,0)
# #         win32api.keybd_event(65,0,win32con.KEYEVENTF_KEYUP,0)
# #         time.sleep(0.1)
# #         win32api.keybd_event(79,0,0,0)
# #         win32api.keybd_event(79,0,win32con.KEYEVENTF_KEYUP,0)
# #         time.sleep(0.1)
# #         win32api.keybd_event(win32con.VK_RETURN, 0, 0, 0)
# #         win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)
# #         time.sleep(0.1)
# #         count = count -1
# #     else:
# #         count = 10
# #         time.sleep(10)
#
# # time.sleep(2)
# # win32api.keybd_event(68,0,0,0)
# # win32api.keybd_event(68,0,win32con.KEYEVENTF_KEYUP,0)
# time.sleep(3)
# while True:
#     win32api.keybd_event(win32con.VK_RETURN,0,0,0)
#     win32api.keybd_event(win32con.VK_RETURN,0,win32con.KEYEVENTF_KEYUP,0)
#     time.sleep(0.5)

a = [1,2,3,4,5,6,7]
action = '1' + '@' + str(a[0]) + ',' + str(a[1]) + ',' + str(a[2]) + ',' + str(a[3]) + ',' + str(a[3]) + ',' + str(a[4]) + ',' + str(a[5]) + ',' + str(a[6])
print(action)