import os
import pickle

transfile_path = '../server/transfile/wkp.txt'
timestamp = None

while True:
    timeinfo = os.stat(transfile_path)
    if timeinfo != timestamp:
        timestamp = timeinfo
        with open(transfile_path,'r') as recv_file:
            trans_file = recv_file.read()
        #trans_file.close()
            print(trans_file)


# #a = [1,2,3,4,5,6,7]
# action = '1' + '@' + str(a[0]) + ',' + str(a[1]) + ',' + str(a[2]) + ',' + str(a[3]) + ',' + str(a[3]) + ',' + str(a[4]) + ',' + str(a[5]) + ',' + str(a[6])
# #print(action)
# self.conn.sendall(action.encode('gb2312'))
# print('successfully send action to client')
# next_obserstr = self.read_trans()
# print('next_obserstr', next_obserstr)
# full_next_obs, next_ourplane, _, _, _ = self.obs_parser(next_obserstr)
# rwd = self.reward(next_obserstr)
# done = 1 if next_ourplane[2] < 0 else 0
# # 关闭客户端连接
# # conn.colse()