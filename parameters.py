import scipy.stats as stats
import os
import numpy as np
import tensorflow as tf
import ipdb as pdb
import matplotlib.pyplot as plt
from options import args_parser

args = args_parser()


# 进行AR信道建模
def complexGaussian(row=1, col=1, amp=1.0):
    real = np.random.normal(size=[row, col])[0] * np.sqrt(0.5)
    # np.random.normal(size=[row,col])生成数据2维，第一维度包含row个数据，每个数据中又包含col个数据
    # np.sqrt(A)求A的开方
    img = np.random.normal(size=[row, col])[0] * np.sqrt(0.5)
    return amp * (real + 1j * img)  # amp相当于系数，后面计算时替换成了根号下1-rou平方    (real + 1j*img)即为误差向量e(t)


class ARModel(object):
    """docstring for AR channel Model"""

    def __init__(self, n_t=1, n_r=1, seed=123):
        self.n_t = n_t
        self.n_r = n_r
        np.random.seed([seed])
        self.H1 = complexGaussian(self.n_t, self.n_r)  # self.H就是hi，即信道增益。初始化定义.

    def sampleCh(self, dis, rho):
        for i in range(args.num_users):
            # self.H1[i] = rho[i] * self.H1[i] + complexGaussian(self.n_t, self.n_r, np.sqrt(1 - rho[i] * rho[i]))  # 这是信道更新的方式
            self.H1[i] = rho[i] * self.H1[i] + complexGaussian(1, 1, np.sqrt(1 - rho[i] * rho[i])) #因为这里是一个一个算的，所以是一个一个复高斯数生成的，所以可以直接写成1,1

        return self.H1


    # def sampleCh(self, rho):
    #     self.H = [0] * args.num_users
    #     for i in range(args.num_users):
    #         self.H[i] = rho[i] * self.H + complexGaussian(self.n_t, self.n_r, np.sqrt(1 - rho[i] * rho[i]))  # rho就是ρi  self.H就是hi（信道增益）
    #     return self.H    # self.H就是hms
