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

from options import args_parser
from local_Update import LocalUpdate, test_inference, get_dataset, average_weights, exp_details, asy_average_weights
from local_model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, cifar_iid, cifar_noniid




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

# load dataset and user groups
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

# 全局模型操作：
glmodel.to(device)
glmodel.train()
print('global model param is:', glmodel.parameters())
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

acc11 = []
total_t = []
AFL_loss1 = []

for ep in tqdm(range(args.epochs)):

    ep_start_time = time.time()
    locweights, locloss = [], []
    print(f'\n | Global Training Round : {ep+1} |\n')

    glmodel.train()

    user_id = range(args.num_users)

    total_time = []

    for j in user_id:
        vehicle_start_time = time.time()
        local_net = copy.deepcopy(vehicle_model[j])
        local_net.to(device)
        locmdl = LocalUpdate(args=args, dataset=trdata, idxs=usgrp[j], logger=logger)
        w, loss, localmodel = locmdl.asyupdate_weights(model=copy.deepcopy(glmodel), global_round=ep, index=j)

        locweights.append(copy.deepcopy(w))
        locloss.append(copy.deepcopy(loss))


        glmodel, glweights = asy_average_weights(vehicle_idx=j, global_model=glmodel, local_model=localmodel,
                                                        gamma=args.gamma)
        # globalmodelw,globalmodelloss,globalmodelmodel = locmdl.asyupdate_weights(model=copy.deepcopy(glmodel), global_round=ep)
        # print("globalmodelloss is ", globalmodelloss)
        # vehicle_model[j] = copy.deepcopy(glmodel)
        # print("locloss is",locloss)
        # avg_step_loss = sum(locloss) / len(locloss)
        # tr_step_loss.append(avg_step_loss)


        # # Calculate avg training accuracy over all users after each user update the model
        # step_acc, step_loss = [], []
        # glmodel.eval()
        # for q in range(args.num_users):
        #     locmdl = LocalUpdate(args=args, dataset=trdata,
        #                          idxs=usgrp[q], logger=logger)
        #     acc1, loss1 = locmdl.inference(model=glmodel)
        #     step_acc.append(acc1)
        #     step_loss.append(loss1)
        # tr_step_acc.append(sum(step_acc) / len(step_acc))
        # tr_step_loss.append(sum(step_loss) / len(step_loss))

        # print step training loss after every 'i' rounds
        # if (ep+1) % print_epoch == 0:
        # print(f' \nAvg Training Stats after {ep + 1} global rounds:')
        # print(f'step Training Loss : {np.mean(np.array(tr_step_loss))}')
        # print("tr_step_loss is", tr_step_loss)
        # print('step Train Accuracy: {:.2f}% \n'.format(100 * tr_step_acc[-1]))

        cost_time = time.time() - vehicle_start_time
        total_time.append(cost_time)
    avg_loss = sum(locloss) / len(locloss)
    print("avg_loss is :", avg_loss)
    AFL_loss1.append(avg_loss)

    print('total_time is:', total_time)
    for iz in range(args.num_users):
        vehicle_model[iz] = copy.deepcopy(glmodel)
    # print(' tr_step_loss is :', tr_step_loss)

    # Test inference after completion of training
    tsacc, tsloss = test_inference(args, glmodel, tsdata)

    print(f' \n Results after {j} epoch rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*tracc[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * tsacc))
    acc11.append(tsacc)

    t1 = time.time() - ep_start_time
    total_t.append(t1)
print('AFL_loss1 is:', AFL_loss1)
print('total_t.append is:', total_t)
print('acc11 is:', acc11)
    # avg_ep_loss = sum(locloss) / len(locloss)
    # trloss.append(avg_ep_loss)

    # Calculate avg training accuracy over all users at every epoch
    # epacc, eploss = [], []
    # glmodel.eval()
    # for q in range(args.num_users):
    #     locmdl = LocalUpdate(args=args, dataset=trdata,
    #                               idxs=usgrp[q], logger=logger)
    #     acc, loss = locmdl.inference(model=glmodel)
    #     epacc.append(acc)
    #     eploss.append(loss)
    # tracc.append(sum(epacc)/len(epacc))

    # print global training loss after every 'i' rounds
    # if (ep+1) % print_epoch == 0:
    # print(f'\nAvg Training Stats after {ep+1} epoch rounds:')
    # print(f'ep Training Loss : {np.mean(np.array(trloss))}')
    # print('ep Train Accuracy: {:.2f}% \n'.format(100*tracc[-1]))

# # Test inference after completion of training
# tsacc, tsloss = test_inference(args, glmodel, tsdata)
#
# print(f' \n Results after {args.epochs} epoch rounds of training:')
# # print("|---- Avg Train Accuracy: {:.2f}%".format(100*tracc[-1]))
# print("|---- Test Accuracy: {:.2f}%".format(100*tsacc))

# Saving the objects train_loss and train_accuracy:
fname = 'results/models/epo_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
           args.local_ep, args.local_bs)

with open(fname, 'wb') as f:
    pickle.dump([trloss, tracc], f)


# Saving the objects train_loss and train_accuracy:
fname = 'results/models/step_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
           args.local_ep, args.local_bs)

with open(fname, 'wb') as f:
    pickle.dump([tr_step_loss, tr_step_acc], f)

print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))