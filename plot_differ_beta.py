
# 不同beta值跑出来的10step acc
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
names = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7','0.8','0.9']
x = range(len(names))
# x = range(9)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字

# y轴的值：
#label=['0.1', '0.2',  '0.3', '0.4','0.5', '0.6', '0.7', '0.8', '0.9']
y_11 = [96.23, 96.30, 96.20, 96.29, 96.42, 95.60, 95.28, 94.32, 91.65]

y_12 =[96.09, 95.96, 96.08, 96.16, 96.08, 95.38, 95.09, 94.17, 91.22]
# plt.plot(x, y_11, color='orangered', marker='o', linestyle='-', label='value of beta')
plt.plot(x, y_11, color='orangered', marker='o', linestyle='-', label='本文方案')
plt.plot(x, y_12, color='blueviolet', marker='D', linestyle='-', label='DDPG选择节点下的传统异步联邦方案')
plt.legend()  # 显示图例
plt.xticks(x, names)
plt.xlabel("β值")  # X轴标签
plt.ylabel("精度")  # Y轴标签
plt.show()