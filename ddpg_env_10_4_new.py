import numpy as np
import ipdb as pdb
import tensorflow as tf
from options import args_parser
from parameters import *
from agent import *

from scipy import special as sp
from scipy.constants import pi

args = args_parser()
args.num_users = 5

# FL相关
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from scipy import special as sp
from scipy.constants import pi

import torch
from tensorboardX import SummaryWriter

from local_Update import LocalUpdate, test_inference, get_dataset, average_weights, exp_details, asy_average_weights
from local_model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, cifar_iid, cifar_noniid


# 进行资源的初始化赋值:
mu1, sigma1 = 1.5e+9, 1e+8
lower, upper = mu1 - 2 * sigma1, mu1 + 2 * sigma1  # 截断在[μ-2σ, μ+2σ]
x = stats.truncnorm((lower - mu1) / sigma1, (upper - mu1) / sigma1, loc=mu1, scale=sigma1)
aaaa = x.rvs(args.num_users)
# print("aaaa is :", aaaa)



class MecTerm(object):
    """
    MEC terminal parent class
    """

    def __init__(self, user_config, train_config):
        self.dis = 0
        self.id = user_config['id']  # 车辆编号
        self.state_dim = user_config['state_dim'] * args.num_users
        self.action_dim = user_config['action_dim']
        self.action_bound = user_config['action_bound']

        self.seed = train_config['random_seed']

        self.tr = [0] * args.num_users
        self.c_c = [0] * args.num_users
        self.l_c = [0] * args.num_users

        self.tr_norm = [0] * args.num_users
        self.delta_norm = [0] * args.num_users
        self.dis_norm = [0] * args.num_users  # state处的归一化

        self.tr_norm1 = [0] * args.num_users
        self.delta_norm1 = [0] * args.num_users
        self.dis_norm1 = [0] * args.num_users  # next_state处的归一化

        # 每辆车的训练数据的样本数d_size （跟FL本地训练分配的本地数据样本数一回事）
        self.d_size = [0] * args.num_users

        alpha = 2
        self.path_loss = [0] * args.num_users
        self.position = [0] * args.num_users
        self.lamda = 7  # 计算ρm时用到的参数
        self.train_config = train_config

        self.init_path = ''
        self.isUpdateActor = True
        self.init_seqCnt = 0

        self.n_t = 1
        self.n_r = user_config['num_r']
        self.P_Lamda = np.zeros(self.action_dim)

        self.Reward = 0
        self.State = []
        self.delta = []
        # some pre-defined parameters
        self.bandwidth = 1000  # Hz   香农公式里的带宽B
        self.velocity = 20.0  # 车辆速度
        self.width_lane = 5  # 车道宽度
        self.Hight_RSU = 10  # RSU高度
        self.transpower = 250  # mw 车辆发送功率（待查具体取值）
        self.sigma2 = train_config['sigma2']  # 噪声功率的平方（后面取10-9mw）

        args.num_users = 5  # 用户数
        self.t = 0.5   # 原0.02

        self.not_reset = True

        # self.channelModel = ARModel(self.n_t, self.n_r, rho=compute_rho(self) ,seed=train_config['random_seed'])
        self.channelModel = ARModel(self.n_t, self.n_r, seed=self.train_config['random_seed'])

    def D_size(self):
        for i in range(args.num_users):
            self.d_size[i] = 1000 + 300 * i
        self.d_size[3] = 100
        return self.d_size

    # 车辆初始坐标位置
    def dis_ini(self):
        self.dis = [0] * args.num_users
        for i in range(args.num_users):  # i从 0 - (args.num_users-1)
            self.dis[i] = -500 + 50 * i  # 车辆初始位置
        return self.dis

    # 车辆移动
    def dis_mov(self):
        # print("dis_mov:")
        if self.not_reset:
            # print("更新车辆位置（成功调用dis_mov函数）")
            for i in range(args.num_users):
                self.dis[i] += (self.velocity * self.t)  # 车辆随时间，x轴行驶距离
        # print("self.dis:", self.dis)
        return self.dis

    def compute_rho(self):
        # print("compute_rho:")
        # 计算ρi
        x_0 = np.array([1, 0, 0])
        P_B = np.array([0, 0, self.Hight_RSU])
        P_m = [0] * args.num_users
        self.rho = [0] * args.num_users
        for i in range(args.num_users):
            P_m[i] = np.array([self.dis[i], self.width_lane, self.Hight_RSU])
            self.rho[i] = sp.j0(2 * pi * self.t * self.velocity * np.dot(x_0, (P_B - P_m[i])) / (
                    np.linalg.norm(P_B - P_m[i]) * self.lamda))
        # print("self.rho:", self.rho)
        return self.rho

    def sampleCh(self, dis):
        # print("sampleCh:")
        self.dis_mov()
        self.compute_rho()
        self.H = self.channelModel.sampleCh(self.dis, self.rho)
        # for i in range(args.num_users):
        #     self.H[i] = self.rho[i] * self.H[i] + complexGaussian(self.n_t, self.n_r, np.sqrt(1 - self.rho[i] * self.rho[i]))  # rho就是ρi  self.H就是hi（信道增益）
        # print("self.H:", self.H)

        # 计算self.path_loss()
        alpha = 2
        self.path_loss = [0] * args.num_users
        self.position = [0] * args.num_users
        for i in range(args.num_users):
            self.position[i] = np.array([self.dis[i], self.width_lane, self.Hight_RSU])  # 车辆的位置坐标
            self.path_loss[i] = 1 / np.power(np.linalg.norm(self.position[i]),
                                             alpha)  # np.linalg.norm(self.position)计算矩阵的模
        # print("self.path_loss:", self.path_loss)


        return self.H, self.path_loss  # self.H就是hms

    # # 车辆与RSU之间距离
    # def Distance(self):
    #     print("Distance:")
    #     self.dis_mov()
    #     alpha = 2
    #     self.path_loss = [0] * args.num_users
    #     self.position = [0] * args.num_users
    #     for i in range(args.num_users):
    #         self.position[i] = np.array([self.dis[i], self.width_lane, self.Hight_RSU])  # 车辆的位置坐标
    #         self.path_loss[i] = 1 / np.power(np.linalg.norm(self.position[i]),
    #                                          alpha)  # np.linalg.norm(self.position)计算矩阵的模
    #     print("self.path_loss:", self.path_loss)
    #     return self.path_loss  # 返回路径损耗di(-α)

    # 车辆和RSU传输速率
    def transRate(self):
        # print("transRate:")
        self.sampleCh(self.dis)
        sinr = [0] * args.num_users

        for i in range(args.num_users):
            sinr[i] = self.transpower * abs(self.H[i]) * self.path_loss[i] / self.sigma2
            # abs()即可求绝对值，也可以求复数的模 #因为神经网络里输入的值不能为复数
            # sinr[i] = self.transpower * self.H[i] * self.path_loss[i] / self.sigma2
            self.tr[i] = np.log2(1 + sinr[i]) * self.bandwidth
        # a = np.reshape(self.tr,(1,args.num_users ))[0]
        # print("self.tr:", self.tr)
        return self.tr

    # 每辆车可用计算资源delta 服从截断高斯分布
    def resou(self):
        # print("resou:")
        mu1, sigma1 = 1.5e+9, 1e+8
        lower, upper = mu1 - 2 * sigma1, mu1 + 2 * sigma1  # 截断在[μ-2σ, μ+2σ]
        x = stats.truncnorm((lower - mu1) / sigma1, (upper - mu1) / sigma1, loc=mu1, scale=sigma1)
        self.delta = x.rvs(args.num_users)  # 总共得到每辆车的计算资源，车辆数目为clients_num来控制
        self.delta[3] = 10
        # print("self.delta:", self.delta)
        return self.delta

    # 所有车的状态集合（即DDPG输入的状态）
    def all_state(self):
        # print("all_state中：")
        self.not_reset = False
        self.transRate()
        self.not_reset = True
        self.State = [0] * args.num_users  # 未整形的state
        # 重置成一样的delta:
        self.delta = aaaa

        # self.tr_norm = [0] * args.num_users    # 前面初始化中写了
        # self.delta_norm = [0] * args.num_users
        # self.dis_norm = [0] * args.num_users

        for i in range(args.num_users):
            self.tr_norm[i] = float(self.tr[i] - np.min(self.tr)) / (np.max(self.tr) - np.min(self.tr))

        for i in range(args.num_users):
            self.delta_norm[i] = float(self.delta[i] - np.min(self.delta)) / (np.max(self.delta) - np.min(self.delta))

        for mm in range(args.num_users):
            # self.dis_norm[i] = float(i - np.min(self.dis))/(np.max(self.dis) - np.min(self.dis))
            self.dis_norm[mm] = float(self.dis[mm] - np.min(self.dis)) / (np.max(self.dis) - np.min(self.dis))

        # for i in range(args.num_users):
        #     self.State[i] = [self.tr[i], self.delta[i], self.dis[i], self.P_Lamda[i]]

        # 归一化
        # for i in range(args.num_users):
        #     self.State[i] = [self.tr_norm[i], self.delta_norm[i], self.dis_norm[i], self.P_Lamda[i]]
        print("重置动作：")

        # self.P_Lamda = np.zeros(5)

        self.State = [self.tr_norm, self.delta_norm, self.dis_norm, self.P_Lamda]
        print("all_state 中 self.State is:", self.State)
        # a = np.reshape(s, (1, 4 * args.num_users))
        # b = np.array(s)
        # c = b.reshape(1, 4 * args.num_users)
        self.State = np.reshape(self.State, (1, 4 * args.num_users))[0]
        return self.State

    # 时隙t车辆i的本地学习时间cost
    def local_c(self):
        self.resou()
        self.D_size()
        beta_m = 1e6  # 执行一个数据样本所需CPU周期数（10^6 cycles）
        # self.l_c = [0] * args.num_users
        for i in range(args.num_users):
            self.l_c[i] = self.d_size[i] * 1e6 / self.delta[i]
        print('self.l_c is:', self.l_c)
        return self.l_c

    # 时隙t车辆i的通信cost,w_i_size为t时隙所学习的本地模型参数的大小，transRate传输速率
    def commu_c(self):
        self.transRate()
        w_size = 5000  # 本地模型参数大小（5kbits）(香农公式传输速率单位为bit/s)
        # self.c_c = [0] * args.num_users
        for i in range(args.num_users):
            self.c_c[i] = w_size / self.tr[i]
        print('self.c_c is:', self.c_c)
        return self.c_c

    # 后面直接用模型训练计算处来的值更新，此处不定义函数


class MecTermLD(MecTerm):
    """
    MEC terminal class for loading from stored models
    """

    def __init__(self, sess, user_config, train_config):
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess

        # 设置模型保存路径等
        saver = tf.train.import_meta_graph(user_config['meta_path'])  # 已将原始网络保存在了.meta文件中，可以用tf.train.import()函数来重建网络
        saver.restore(sess, user_config['model_path'])

        graph = tf.get_default_graph()  # 获取当前默认计算图
        input_str = "input_" + self.id + "/X:0"
        output_str = "output_" + self.id + ":0"
        self.inputs = graph.get_tensor_by_name(input_str)
        if not 'action_level' in user_config:
            self.out = graph.get_tensor_by_name(output_str)

    def feedback(self, P_lamda, AFL_loss, PP_lamda1):  # AFL_loss 为每一个step后的全局loss. PP_lamda1为离散后的动作列表
        # self.P_Lamda = P_lamda
        self.next_state = []

        # 更新下一状态以及需要计算奖励相关的函数
        # print("self.local_c()前的 self.l_c is :", self.l_c)
        self.local_c()  # 里面有delta的更新，所以不需要self.resou()
        # print("后的 self.l_c is :", self.l_c)
        # print("这里是feedback里面更新下一状态之前")
        self.commu_c()  # 这里面的车辆位置更新了
        # print("这里是feedback里面更新下一状态之后")
        # print("self.dis:", self.dis)
        # print("self.path_loss:", self.path_loss)

        # update the transmission rate
        # self.transRate()

        # get the reward for the current slot
        # aa = []
        # for i1, j1 in zip(self.l_c, self.c_c):
        #     summ = i1 + j1
        #     aa.append(summ)

        # eee = self.l_c + self.c_c + self.globall
        # # eee = []
        # # for ii, jj in zip(aa, self.globall):
        # #     summe = ii + jj
        # #     eee.append(summe)
        #
        # bb = np.multiply(eee, self.P_Lamda)
        # # bb = []
        # # for iii, jjj in zip(eee, self.P_Lamda):
        # #     cc = iii * jjj
        # #     bb.append(cc)
        # self.Reward = -sum(bb)   # 对应奖励公式

        a1 = np.array(self.l_c)
        # print("a1=self.l_c=", a1)
        b1 = np.array(self.c_c)
        # print("b1=self.c_c=", b1)
        c1 = a1 + b1
        # print("c1=a1 + b1=", c1)
        bb = np.multiply(c1, PP_lamda1)
        # print(" bb = np.multiply(c1, PP_lamda1) = ", bb)
        print("AFL_loss is :", AFL_loss)
        print("sum(bb) / sum(PP_lamda1) :", sum(bb) / sum(PP_lamda1))
        # print("localloss:", localloss)
        # print("self.P_Lamda * localloss:", self.P_Lamda * localloss)
        # print("sum(bb):", sum(bb))
        # print("sum(self.P_Lamda * localloss)", sum(self.P_Lamda * localloss))
        # self.Reward = (-sum(bb) - 50*AFL_loss) / sum(PP_lamda1)
        self.Reward = -(AFL_loss + sum(bb) / sum(PP_lamda1) / 3) / (sum(self.P_Lamda)/len(self.P_Lamda))
        coststep = sum(bb) / sum(PP_lamda1)

        # print("self.Reward = -AFL_loss - sum(bb) / sum(PP_lamda1) / 3")
        # print("step reward is :", self.Reward)

        ######
        # np.argmax(self.P_Lamda)为动作中最大值的索引
        # self.Reward = - (self.l_c[np.argmax(self.P_Lamda)] + self.c_c[np.argmax(self.P_Lamda)] + self.globall) * max(
        #     self.P_Lamda) / sum(self.P_Lamda)

        # update the actor and critic network
        # 对状态进行归一化处理
        for i in range(args.num_users):
            self.tr_norm1[i] = float(self.tr[i] - np.min(self.tr)) / (np.max(self.tr) - np.min(self.tr))

        for i in range(args.num_users):
            self.delta_norm1[i] = float(self.delta[i] - np.min(self.delta)) / (
                        np.max(self.delta) - np.min(self.delta))

        for i in range(args.num_users):
            self.dis_norm1[i] = float(self.dis[i] - np.min(self.dis)) / (np.max(self.dis) - np.min(self.dis))

        # self.next_state = np.array([self.tr_norm1, self.delta_norm1, self.dis_norm1, self.P_lamda])  # 定义下一状态更新
        self.next_state = np.array([self.tr_norm1, self.delta_norm1, self.dis_norm1, P_lamda])  # 定义下一状态更新
        # print("self.next_state is :", self.next_state)

        # update system state
        self.State = self.next_state
        # return the reward in this slot
        # return self.Reward, self.tr, self.delta, self.dis, self.P_lamda
        return self.Reward, self.tr, self.delta, self.dis, P_lamda, coststep
    def predict(self, isRandom):
        self.P_Lamda = self.sess.run(self.out, feed_dict={self.inputs: np.reshape(self.State, (1, self.state_dim))})[0]
        return self.P_Lamda

    """
        a = tf.add(2, 5)
        b = tf.multiply(a, 3)
        with tf.Session() as sess: 
	    sess.run(b, feed_dict = {a:15}) # 重新给a赋值为15   运行结果：45
	    sess.run(b) #feed_dict只在调用它的方法内有效,方法结束,feed_dict就会消失。 所以运行结果是：21   
    """


class MecTermRL(MecTerm):
    """
    MEC terminal class using RL
    """

    # rate:packet poisson arrival, dis: distance in meters
    def __init__(self, sess, user_config, train_config):
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess
        self.agent = DDPGAgent(sess, user_config, train_config)

        if 'init_path' in user_config and len(user_config['init_path']) > 0:
            self.init_path = user_config['init_path']
            self.init_seqCnt = user_config['init_seqCnt']
            self.isUpdateActor = False

    def feedback(self, P_lamda, AFL_loss, PP_lamda1):  # AFL_loss 为每一个step后的全局loss. PP_lamda1为离散后的动作列表
        self.P_Lamda = P_lamda
        self.next_state = []

        # 更新下一状态以及需要计算奖励相关的函数
        # print("self.local_c()前的 self.l_c is :", self.l_c)
        self.local_c()  # 里面有delta的更新，所以不需要self.resou()
        # print("后的 self.l_c is :", self.l_c)
        # print("这里是feedback里面更新下一状态之前")
        self.commu_c()  # 这里面的车辆位置更新了
        # print("这里是feedback里面更新下一状态之后")
        # print("self.dis:", self.dis)
        # print("self.path_loss:", self.path_loss)


        # update the transmission rate
        # self.transRate()

        # get the reward for the current slot
        # aa = []
        # for i1, j1 in zip(self.l_c, self.c_c):
        #     summ = i1 + j1
        #     aa.append(summ)

        # eee = self.l_c + self.c_c + self.globall
        # # eee = []
        # # for ii, jj in zip(aa, self.globall):
        # #     summe = ii + jj
        # #     eee.append(summe)
        #
        # bb = np.multiply(eee, self.P_Lamda)
        # # bb = []
        # # for iii, jjj in zip(eee, self.P_Lamda):
        # #     cc = iii * jjj
        # #     bb.append(cc)
        # self.Reward = -sum(bb)   # 对应奖励公式

        a1 = np.array(self.l_c)
        print("a1=self.l_c=", a1)
        b1 = np.array(self.c_c)
        print("b1=self.c_c=", b1)
        c1 = a1 + b1
        print("c1=a1 + b1=", c1)
        bb = np.multiply(c1, PP_lamda1)
        print(" bb = np.multiply(c1, PP_lamda1) = ", bb)
        print("sum(bb) / sum(PP_lamda1) :", sum(bb) / sum(PP_lamda1))
        # print("localloss:", localloss)
        # print("self.P_Lamda * localloss:", self.P_Lamda * localloss)
        # print("sum(bb):", sum(bb))
        # print("sum(self.P_Lamda * localloss)", sum(self.P_Lamda * localloss))
        # self.Reward = (-sum(bb) - 50*AFL_loss) / sum(PP_lamda1)
        self.Reward = -(AFL_loss + sum(bb) / sum(PP_lamda1) / 3) / (sum(self.P_Lamda)/len(self.P_Lamda))
        print("self.Reward = -AFL_loss - sum(bb) / sum(PP_lamda1) / 3")
        print("step reward is :", self.Reward)

        ######
        # np.argmax(self.P_Lamda)为动作中最大值的索引
        # self.Reward = - (self.l_c[np.argmax(self.P_Lamda)] + self.c_c[np.argmax(self.P_Lamda)] + self.globall) * max(
        #     self.P_Lamda) / sum(self.P_Lamda)


        # update the actor and critic network
        # 对状态进行归一化处理
        for i in range(args.num_users):
            self.tr_norm1[i] = float(self.tr[i] - np.min(self.tr)) / (np.max(self.tr) - np.min(self.tr))

        for i in range(args.num_users):
            self.delta_norm1[i] = float(self.delta[i] - np.min(self.delta)) / (np.max(self.delta) - np.min(self.delta))

        for i in range(args.num_users):
            self.dis_norm1[i] = float(self.dis[i] - np.min(self.dis)) / (np.max(self.dis) - np.min(self.dis))

        self.next_state = np.array([self.tr_norm1, self.delta_norm1, self.dis_norm1, self.P_lamda])  # 定义下一状态更新
        print("self.next_state is :", self.next_state)
        # print("这句话前面")
        # self.next_state = np.array([self.tr, self.delta, self.dis, self.P_lamda])  # 定义下一状态更新
        # print("执行这句话了")
        # update system state
        # self.State = self.next_state

        # return the reward in this slot
        # return self.Reward, self.tr, self.delta, self.dis, self.P_lamda
        return self.Reward, self.tr_norm1, self.delta_norm1, self.dis_norm1, self.P_lamda

    # LD里的
    # def predict(self, isRandom):
    #     self.P_Lamda = self.sess.run(self.out, feed_dict={self.inputs: np.reshape(self.State, (1, self.state_dim))})[0]
    #     return self.P_Lamda, np.zeros(self.action_dim)

    def predict(self, isRandom):
        P_lamda1 = self.agent.predict(self.State, self.isUpdateActor)
        self.P_lamda = np.fmax(0, np.fmin(self.action_bound, P_lamda1))
        return self.P_lamda

    def AgentUpdate(self, done):
        # 往replaybuffer加动作状态，到达一定数量更新它和神经网络：
        print("self.isUpdateActor is :", self.isUpdateActor)
        self.agent.update(self.State, self.P_lamda, self.Reward, done, self.next_state, self.isUpdateActor)
        print("self.State is :", self.State)
        print("self.next_state is:", self.next_state)
        self.State = self.next_state


class MecSvrEnv(object):
    """
    Simulation environment
    """

    def __init__(self, user_list, Train_vehicle_ID, sigma2, max_len, mode='train'):
        self.user_list = user_list
        self.Train_vehicle_ID = Train_vehicle_ID - 1
        self.sigma2 = sigma2
        self.mode = mode   # 训练or测试
        self.count = 0
        self.max_len = max_len

    def init_target_network(self):
        self.user_list[self.Train_vehicle_ID].agent.init_target_network() # 即为RL里面的agent, 即为DDPGagent.

    def step_transmit(self, isRandom=True):
        # the id of vehicle for training
        i = self.Train_vehicle_ID

        P_lamda1 = self.user_list[i].predict(isRandom)

        rewards = 0
        trs = 0
        deltas = 0
        diss = 0

        self.count += 1

        # feedback the sinr to each user
        [rewards, trs, deltas, diss, P_lamdas] = self.user_list[i].feedback(P_lamda1)

        if self.mode == 'train':
            self.user_list[i].AgentUpdate(self.count >= self.max_len)  # 训练数据个数逐渐增加，大于buffer的大小时，进行更新agent

        return rewards, self.count >= self.max_len, P_lamdas, trs, deltas, diss  # self.count >= self.max_len对应的是训练里面的done

    def reset(self, isTrain=True):  # 将所有环境变量全部重置（一个大的episode重置一次）  注：车辆数据大小不变
        # the id of vehicle for training
        i = self.Train_vehicle_ID
        self.count = 0
        # 车辆位置重置
        print("env.reset 中初始化车辆位置：")
        for user in self.user_list:
            if self.mode == 'train':
                user.dis_ini()
            elif self.mode == 'test':
                user.dis_ini()

        print("初始化完车辆位置")
        # 信道不用专门重置了，因为重置状态里已经重置过了
        # get the channel vectors  信道信息重置
        # channels = [user.Distance() for user in self.user_list]

        # 这里，5为用户个数，更改用户个数时这里也要改动


        # 重置状态
        print("重置所有状态：")
        self.user_list[i].all_state()
        print("重置完所有状态")
        # self.user_list[i].dis_mov()

        return self.count  # 不返回都行，重要的是环境重置了
