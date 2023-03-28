import numpy as np
# AFL_weight vs AFL : acc对比
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

names = ['1', '2', '3', '4', '5', '6', '7','8','9','10']
x = range(len(names))
# x = range(,10)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字


y_14 = [0.8412, 0.9219, 0.9398, 0.9465, 0.9528, 0.9543, 0.9578, 0.9579, 0.9634, 0.9642]  # AFL_weight
y_24 = [0.8369, 0.9178, 0.9311, 0.9379, 0.9475, 0.9518, 0.9558, 0.9545, 0.9612, 0.9608]  # AFL
y_34 = [0.6397, 0.8527, 0.8952, 0.9092, 0.9276, 0.9333, 0.9424, 0.9480, 0.9501, 0.9525]  # FL
plt.plot(x, y_14, color='orangered', marker='o', linestyle='-', label='本文方案')
plt.plot(x, y_24, color='blueviolet', marker='D', linestyle='-.', label='DDPG选择节点下的传统异步联邦方案')
plt.plot(x, y_34, color='green', marker='*', linestyle=':', label='DDPG选择节点下的传统联邦方案')
plt.legend()  # 显示图例
# plt.xticks(x, names, rotation=45)
# plt.xticks(x, names)
plt.xticks(x, names)
plt.xlabel("步数")  # X轴标签
plt.ylabel("精度")  # Y轴标签
plt.show()



y_1 = [1.7168, 0.8660, 0.6458, 0.5460, 0.4765, 0.4429, 0.4130, 0.3918, 0.3667, 0.3423]  # AFL_weight
y_2 = [1.7817, 0.9289, 0.6980, 0.5896, 0.5155, 0.4586, 0.4225, 0.3927, 0.3701, 0.3477] # AFL
y_3 = [2.1304, 1.4598, 1.0218, 0.8407, 0.7039, 0.6588, 0.5748, 0.5509, 0.4993, 0.4604]  # FL
plt.plot(x, y_1, color='orangered', marker='o', linestyle='-', label='本文方案')
plt.plot(x, y_2, color='blueviolet', marker='D', linestyle='-.', label='DDPG选择节点下的传统异步联邦方案')
plt.plot(x, y_3, color='green', marker='*', linestyle=':', label='DDPG选择节点下的传统联邦方案')
plt.legend()  # 显示图例
# plt.xticks(x, names, rotation=45)
# plt.xticks(x, names)
plt.xticks(x, names)
plt.xlabel("步数")  # X轴标签
plt.ylabel("损失")  # Y轴标签
plt.show()

# AFL_weight:
a1 = [1.7155, 0.9080, 0.6600, 0.5726, 0.4756, 0.4530, 0.4235, 0.3848, 0.3684, 0.3580]
a2 = [1.7124, 0.8774, 0.6427, 0.5610, 0.4747, 0.4420, 0.4306, 0.3693, 0.3598, 0.3386]
a3 = [1.7046, 0.8309, 0.6545, 0.5375, 0.4851, 0.4549, 0.3981, 0.4048, 0.3927, 0.3492]
a4 = [1.7336, 0.8897, 0.6404, 0.5396, 0.4697, 0.4320, 0.4105, 0.4013, 0.3478, 0.3392]
a5 = [1.8173, 0.9249, 0.6644, 0.5392, 0.4774, 0.4334, 0.4031, 0.3860, 0.3649, 0.3534]
a = np.array(a1)+np.array(a2)+np.array(a3)+np.array(a4)+np.array(a5)

# weight_less:
b1 = [1.6929, 0.8737, 0.6890, 0.5863, 0.5103, 0.4590, 0.4026, 0.3870, 0.3587, 0.3331]
b2 = [1.6848, 0.9131, 0.6761, 0.5857, 0.4957, 0.4603, 0.4266, 0.3930, 0.3744, 0.3453]
b3 = [1.8576, 0.9638, 0.7457, 0.6215, 0.5473, 0.4919, 0.4502, 0.4235, 0.3906, 0.3875]
b4 = [1.8291, 0.9362, 0.6836, 0.5649, 0.4872, 0.4216, 0.4048, 0.3796, 0.3551, 0.3342]
b5 = [1.8445, 0.9580, 0.6959, 0.5897, 0.5371, 0.4603, 0.4285, 0.3806, 0.3720, 0.3387]

b = np.array(b1)+np.array(b2)+np.array(b3)+np.array(b4)+np.array(b5)


