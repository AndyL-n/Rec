# import json
# import requests
# import time
# from openpyxl import load_workbook
# wb = load_workbook('./电费.xlsx')
# ws = wb.active
# ws.cell(1,1).value = '房间号'
# ws.cell(1,2).value = '总电量'
# ws.cell(1,3).value = '用电量'
# ws.cell(1,4).value = '余电量'
# ws.cell(1,5).value = '更新时间'
# ws.cell(1,6).value = '当前时间'
# ws.title = "电费"
# wb.save('./电费.xlsx')
# flag = 1
# while(1):
#     date = time.localtime(time.time())
#     if (date.tm_sec == flag):
#         continue
#     else:
#         postUrl = 'https://kdkd.sizhu.tech/api/electricity/init'
#         loadData = { "open_id": "oUiRowbIP28uEIidRm4JEYrriU0k" }
#         loadHeader = {
#         'Host': 'kdkd.sizhu.tech',
#         'Content-Type': 'application/json;charset=UTF-8',
#         }
#         r = requests.post(postUrl, data=json.dumps(loadData), headers=loadHeader).json()
#         # print("房间号：\t" + r['data']['room'])
#         # print("总电量：\t" + str(r['data']['allAmp']))
#         # print("用电量：\t" + str(r['data']['usedAmp']))
#         # print("余电量：\t" + str(r['data']['allAmp'] - r['data']['usedAmp']))
#         # print("更新时间:"+ r['data']['updateAt'])
#         # print("当前时间:"+ str(date.tm_year) + "-" +("0" + str(date.tm_mon) if date.tm_mon < 10 else str(date.tm_mon)) + "-" + ("0" + str(date.tm_mday) if date.tm_mday < 10 else str(date.tm_mday)) + " " + ("0" + str(date.tm_hour) if date.tm_hour < 10 else str(date.tm_hour)) + ":" + ("0" + str(date.tm_min) if date.tm_min < 10 else str(date.tm_min)) + ":" + ("0" + str(date.tm_sec) if date.tm_sec < 10 else str(date.tm_sec)))
#         index = ws.max_row + 1
#         ws.cell(index, 1).value = r['data']['room']
#         ws.cell(index, 2).value = (r['data']['allAmp'])
#         ws.cell(index, 3).value = (r['data']['usedAmp'])
#         ws.cell(index, 4).value = (r['data']['allAmp'] - r['data']['usedAmp'])
#         ws.cell(index, 5).value = r['data']['updateAt']
#         ws.cell(index, 6).value = str(date.tm_year) + "-" +("0" + str(date.tm_mon) if date.tm_mon < 10 else str(date.tm_mon)) + "-" + ("0" + str(date.tm_mday) if date.tm_mday < 10 else str(date.tm_mday)) + " " + ("0" + str(date.tm_hour) if date.tm_hour < 10 else str(date.tm_hour)) + ":" + ("0" + str(date.tm_min) if date.tm_min < 10 else str(date.tm_min)) + ":" + ("0" + str(date.tm_sec) if date.tm_sec < 10 else str(date.tm_sec))
#         print(ws.max_row)
#         print("房间号：\t" + r['data']['room'], end="\t")
#         print("总电量：\t" + str(r['data']['allAmp']), end="\t")
#         print("用电量：\t" + str(r['data']['usedAmp']), end="\t")
#         print("余电量：\t" + str(r['data']['allAmp'] - r['data']['usedAmp']), end="\t")
#         print("更新时间:"+ r['data']['updateAt'], end="\t")
#         print("当前时间:"+ str(date.tm_year) + "-" +("0" + str(date.tm_mon) if date.tm_mon < 10 else str(date.tm_mon)) + "-" + ("0" + str(date.tm_mday) if date.tm_mday < 10 else str(date.tm_mday)) + " " + ("0" + str(date.tm_hour) if date.tm_hour < 10 else str(date.tm_hour)) + ":" + ("0" + str(date.tm_min) if date.tm_min < 10 else str(date.tm_min)) + ":" + ("0" + str(date.tm_sec) if date.tm_sec < 10 else str(date.tm_sec)))
#         wb.save('./电费.xlsx')
#         flag = date.tm_sec

import tensorflow.keras as keras
import tensorflow as tf
# print(keras.__version__)
print(tf.__version__)