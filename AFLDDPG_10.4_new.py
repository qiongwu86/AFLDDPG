# FL相关

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # -1表示不使用GPU   0/1为显卡名称（使用哪个显卡） 后面联邦学习里面设置了
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

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# DDPG相关

from ddpg_env_10_4_new import *
from parameters import *
from agent import *
import tensorflow as tf

import tflearn
import ipdb as pdb
from options import *

# AFL相关
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

start_time = time.time()

# define paths
path_project = os.path.abspath('..')
logger = SummaryWriter('../logs')

args = args_parser()
exp_details(args)

if args.gpu == 1:
    torch.cuda.set_device(0)
device = 'cuda' if args.gpu else 'cpu'




tf.compat.v1.reset_default_graph()
MAX_EPISODE = 2
MAX_EPISODE_LEN = 5
NUM_R = args.num_users

SIGMA2 = 1e-9
args.num_users = 5


noise_sigma = 0.05

config = {'state_dim': 4, 'action_dim': args.num_users};
train_config = {'minibatch_size': 64, 'actor_lr': 0.0001, 'tau': 0.001,
                'critic_lr': 0.001, 'gamma': 0.99, 'buffer_size': 250000,
                'random_seed': int(time.perf_counter() * 1000 % 1000), 'noise_sigma': noise_sigma, 'sigma2': SIGMA2}

IS_TRAIN = False

res_path = 'train/'
model_fold = 'model/'
model_path = 'model/train_model_-2000'

if not os.path.exists(res_path):
    os.mkdir(res_path)
if not os.path.exists(model_fold):
    os.mkdir(model_fold)

init_path = ''


Train_vehicle_ID = 1


user_config = [{'id': '1', 'model': 'AR', 'num_r': NUM_R, 'action_bound': 1}]

# 0. initialize the session object
sess = tf.compat.v1.Session()

user_list = [];
for info in user_config:
    info.update(config)
    info['model_path'] = model_path
    info['meta_path'] = info['model_path'] + '.meta'
    info['init_path'] = init_path
    user_list.append(MecTermRL(sess, info, train_config))

    print('Initialization OK!----> user ')


env = MecSvrEnv(user_list, Train_vehicle_ID, SIGMA2, MAX_EPISODE_LEN)

sess.run(tf.compat.v1.global_variables_initializer())

tflearn.config.is_training(is_training=IS_TRAIN, session=sess)

env.init_target_network()

res_r = []
res_p = []




# 开始训练episode
for ep in tqdm(range(MAX_EPISODE)):

    trdata, tsdata, usgrp = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            glmodel = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            glmodel = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            glmodel = CNNCifar(args=args)
    elif args.model == 'mlp':
        imsize = trdata[0][0].shape
        input_len = 1
        for x in imsize:
            input_len *= x
            glmodel = MLP(dim_in=input_len, dim_hidden=64,
                          dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    glmodel.to(device)
    glmodel.train()

    vehicle_model = []
    for i in range(args.num_users):
        vehicle_model.append(copy.deepcopy(glmodel))

    # copy weights
    glweights = glmodel.state_dict()

    # Training
    trloss, tracc = [], []
    tr_step_loss = []
    tr_step_acc = []
    vlacc, net_ = [], []
    cvloss, cvacc = [], []
    print_epoch = 2
    vllossp, cnt = 0, 0


    print(f'\n | Global Training Round/episode : {ep + 1} |\n')
    glmodel.train()
    user_id = range(args.num_users)


    plt.ion()
    cur_init_ds_ep = env.reset()

    count = 0
    cur_r_ep = 0

    cur_p_ep = [0] * args.num_users
    step_cur_r_ep = []


    tsacc1 = []
    print("the number of episode(ep):", ep)

    for j in range(MAX_EPISODE_LEN):

        i = Train_vehicle_ID - 1

        # pri = MecTermRL(sess, info, train_config)

        print("the j-th step, j=", j)


        P_lamda1 = user_list[i].predict(True)
        print("P_lamda1 is:", P_lamda1)
        P_lamda11 = [0] * args.num_users

        for qqq in range(args.num_users):
            P_lamda11[qqq] = float(P_lamda1[qqq] - np.min(P_lamda1)) / (np.max(P_lamda1) - np.min(P_lamda1))
        print("归一化后的P_lamda1即P_lamda11为：",  P_lamda11)


        locloss = []
        for aa in user_id:
            if P_lamda11[aa] >= 0.5:
                local_net = copy.deepcopy(vehicle_model[aa])
                local_net.to(device)
                locmdl = LocalUpdate(args=args, dataset=trdata, idxs=usgrp[aa], logger=logger)
                w, loss, localmodel = locmdl.asyupdate_weights(model=copy.deepcopy(glmodel), global_round=ep)
                # locloss[aa] = copy.deepcopy(loss)
                locloss.append(loss)
                # print("locloss :", locloss)
                # print("finished computing loss with vehicle:",aa)


                glmodel, glweights = asy_average_weights(vehicle_idx=aa, global_model=glmodel, local_model=localmodel,
                                                         gamma=args.gamma)
                print("vehicle aa has updated global model: aa 为：", aa)
        print("locloss is :", locloss)
        avg_loss = sum(locloss) / len(locloss)
        print("avg_loss is :", avg_loss)

        for iz in range(args.num_users):
            vehicle_model[iz] = copy.deepcopy(glmodel)
        print("all vehicles have got the updated glmodel")


        PP_lamda = [0] * args.num_users
        for ia in range(args.num_users):
            if P_lamda11[ia] >= 0.5:
                PP_lamda[ia] = 1
        print("离散化后的动作即PP_lamda is :", PP_lamda)


        tsacc, tsloss = test_inference(args, glmodel, tsdata)

        tsacc1.append(tsacc)


        rewards = 0
        trs = 0
        deltas = 0
        diss = 0

        count += 1

        [rewards, trs, deltas, diss, P_lamdas] = user_list[i].feedback(P_lamda1, avg_loss, PP_lamda)

        # print("reward is:", rewards)
        max_len = MAX_EPISODE_LEN
        mode = 'train'
        if mode == 'train':
            user_list[i].AgentUpdate(count >= max_len)
        cur_r = rewards
        cur_p = P_lamdas
        done = count >= max_len

        cur_r_ep += cur_r



    plt.figure()
    plt.title('acc of each epoch')
    plt.plot(range(MAX_EPISODE_LEN), tsacc1, color='b')  # 横轴是epi数，纵轴是每个epi中平均每一step的奖励（每一步的平均奖励）
    plt.ylabel('acc')
    plt.xlabel('Num of steps')
    plt.savefig('acc_{}_ep={}.png'.format(time.time(), ep))



    res_r.append(cur_r_ep / MAX_EPISODE_LEN)

    print("epoch = ", ep)
    print("r = ", cur_r_ep / MAX_EPISODE_LEN)

# Test inference after completion of training
tsacc_final, tsloss_final = test_inference(args, glmodel, tsdata)

print(f' \n Results after {MAX_EPISODE} epoch rounds of training:')
# print("|---- Avg Train Accuracy: {:.2f}%".format(100*tracc[-1]))
print("|---- Test Accuracy: {:.2f}%".format(100 * tsacc_final))

# Saving the objects train_loss and train_accuracy:
fname = 'results/models/epo_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
    format(args.dataset, args.model, MAX_EPISODE, args.frac, args.iid,
           args.local_ep, args.local_bs)

with open(fname, 'wb') as f:
    pickle.dump([trloss, tracc], f)

# Saving the objects train_loss and train_accuracy:
fname = 'results/models/step_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_{}.pkl'. \
    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
           args.local_ep, args.local_bs, time.localtime(time.time()))

with open(fname, 'wb') as f:
    pickle.dump([tr_step_loss, tr_step_acc], f)

print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

# DDPG模型保存
name = res_path + 'DDPG_model' + time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime(time.time()))
np.savez(name, res_r)

tflearn.config.is_training(is_training=False, session=sess)
# Create a saver object which will save all the variables
saver = tf.compat.v1.train.Saver()
saver.save(sess, model_path)
sess.close()

# Plot curve
plt.figure()
plt.title('epoch reward')
plt.plot(range(MAX_EPISODE), res_r, color='b')
plt.ylabel('reward')
plt.xlabel('Num of epochs')
plt.savefig('reward_{}.png'.format(time.time()))
