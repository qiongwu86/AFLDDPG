#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import numpy as np
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, cifar_iid, cifar_noniid
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import tensorflow as tf
sess = tf.Session()


def modelRecovery(state_dict, model):
    return model.load_state_dict(state_dict, strict=True)

def modelSnapshot(model):
    return model.state_dict()


def get_dataset(args):
    if args.dataset == 'cifar':
        data_dir = 'data/cifar/'  #MNIST
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = 'data/mnist/'
        else:
            data_dir = 'data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose unequal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose equal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def asy_average_weights(vehicle_idx, global_model, local_model, gamma):
    print("vehicle ", vehicle_idx+1, " has already updated the RSU!")


    for name, param in global_model.named_parameters():
        for name2, param2 in local_model.named_parameters():
            if name == name2:
                param.data.copy_(gamma * param.data + (1 - gamma) * param2.data)
                # param.data.copy_(param.data + param2.data)
    # print("Update decayed by factor", local_update_param)
    return global_model, global_model.state_dict()



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return image.clone().detach().requires_grad_(True), label.clone().detach().requires_grad_(True)
        return torch.tensor(image), torch.tensor(label)
# To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach()
# or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)



    def train_val_test(self, dataset, idxs):
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, index):
        # Set mode to train model
        model.train()  # 设置模型为训练模式
        epoch_loss = []

        # 模型加噪声：
        if index == 3:
            for parameters in model.parameters():
                with torch.no_grad():
                    model_noise = np.random.normal(0.01,0.05,1)
                    parameters += model_noise[0]

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)

        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # 梯度
                model.zero_grad()
                # 训练预测
                log_probs = model(images)
                # 计算损失函数
                loss = self.criterion(log_probs, labels)     # 损失函数 self.criterion(output, target)
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer.step()

                # 打印本地样本训练过程（进度）
                # if self.args.verbose and (batch_idx % 10 == 0):
                #
                #     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         global_round, iter, batch_idx * len(images),
                #         len(self.trainloader.dataset),
                #         100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))



        return model.state_dict(), sum(epoch_loss) / len(epoch_loss) , model
        # state_dict作用：保存模型中的weight权值和bias偏置值

    # def asyupdate_weights(self, model, global_round, index):
    def asyupdate_weights(self, model, global_round, index):
        # Set mode to train model
        model.train()  # 设置模型为训练模式
        epoch_loss = []
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)

        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        # for iter in range(self.args.local_ep):
        #     batch_loss = []
        #     state_dict = modelSnapshot(model)
        #     for batch_idx, (images, labels) in enumerate(self.trainloader):
        #         images, labels = images.to(self.device), labels.to(self.device)
        #         # 梯度
        #         model.zero_grad()
        #
        #         # 正常客户：
        #         if index != 0:
        #         # 训练预测( output = model(input) )
        #             log_probs = model(images)
        #             # 计算损失函数
        #             loss = self.criterion(log_probs, labels)     # 损失函数 self.criterion(output, target)
        #
        #             # 反向传播,得到每一个要更新的参数的梯度
        #             loss.backward()
        #             # 更新参数(每一个参数都会根据反向传播得到的梯度进行优化)
        #             optimizer.step()
        #
        #         # classflip:
        #         if index == 0:
        #             # 训练预测( output = model(input) )
        #             log_probs = model(images)
        #             # 计算损失函数
        #             loss = self.criterion(log_probs, 9-labels)  # 损失函数 self.criterion(output, target)
        #
        #             # 反向传播,得到每一个要更新的参数的梯度
        #             loss.backward()
        #             # 更新参数(每一个参数都会根据反向传播得到的梯度进行优化)
        #             optimizer.step()
        #
        #             modelRecovery(state_dict, model)
        #
        #         # # dataflip:
        #         # if index == 0:
        #         #     # 训练预测( output = model(input) )
        #         #     log_probs = model(1-images)
        #         #     # 计算损失函数
        #         #     loss = self.criterion(log_probs, labels)  # 损失函数 self.criterion(output, target)
        #         #     # 反向传播,得到每一个要更新的参数的梯度
        #         #     loss.backward()
        #         #     # 更新参数(每一个参数都会根据反向传播得到的梯度进行优化)
        #         #     optimizer.step()
        #
        #
        #         self.logger.add_scalar('loss', loss.item())
        #         batch_loss.append(loss.item())
        #     epoch_loss.append(sum(batch_loss)/len(batch_loss))
        #
        # # # 模型加噪声：
        # # if index == 3:
        # #     for parameters in model.parameters():
        # #         with torch.no_grad():
        # #             model_noise = np.random.normal(0.01,0.08,1)
        # #             parameters += model_noise[0]
        #
        #

        # if index != 4 and index != 2 and index !=0:

        if index != 4 and index != 2:
            for iter in range(self.args.local_ep):
                batch_loss = []

                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    # 梯度
                    model.zero_grad()

                # 训练预测( output = model(input) )
                    log_probs = model(images)
                    # 计算损失函数
                    loss = self.criterion(log_probs, labels)     # 损失函数 self.criterion(output, target)

                    # 反向传播,得到每一个要更新的参数的梯度
                    loss.backward()
                    # 更新参数(每一个参数都会根据反向传播得到的梯度进行优化)
                    optimizer.step()

                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # if index == 0 or index ==1:
        if index ==2 or index ==4:
            for iter in range(self.args.local_ep):
                batch_loss = []

                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    # 梯度
                    model.zero_grad()

                    # 训练预测( output = model(input) )
                    log_probs = model(1-images)
                    # 计算损失函数
                    loss = self.criterion(log_probs, labels)  # 损失函数 self.criterion(output, target)

                    # 反向传播,得到每一个要更新的参数的梯度
                    loss.backward()
                    # 更新参数(每一个参数都会根据反向传播得到的梯度进行优化)
                    # optimizer.step()

                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))




        # # 模型加噪声：
        # if index == 3:
        #     for parameters in model.parameters():
        #         with torch.no_grad():
        #             model_noise = np.random.normal(0.01,0.08,1)
        #             parameters += model_noise[0]



        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), model
        # state_dict作用：保存模型中的weight权值和bias偏置值






    def inference(self, model):

        model.eval()  # 设置模型为评估/推理模式
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            # 聚合所有的损失
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            # 统计预测结果与真实标签target的匹配总个数
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return



def asy_average_weights(vehicle_idx,global_model,local_model,gamma):
    #print("vehicle ", vehicle_idx+1, " has already updated the RSU!")
    for name, param in global_model.named_parameters():
        for name2, param2 in local_model.named_parameters():
            if name == name2:
                param.data.copy_(gamma * param.data + (1 - gamma) * param2.data)

    #print("Update decayed by factor")
    return global_model, global_model.state_dict()

def asy_average_weights_weight(vehicle_idx, global_model, local_model, gamma, local_param2, local_param3):
    # print("vehicle ", vehicle_idx + 1, " has already updated the RSU!")

    for name, param in global_model.named_parameters():
        for name2, param2 in local_model.named_parameters():
            if name == name2:
                param.data.copy_(gamma * param.data + (1 - gamma)  * local_param2 * local_param3 * param2.data)
                # param.data.copy_(param.data + local_param1 * local_param2 * local_param3 * param2.data)

    # print('#######################')
    # print("Update decayed by factor beta_lt[i] , c_c[i]:", local_param2, local_param3)
    # print("beta_lt[i] * c_c[i] is :", local_param2 * local_param3)
    # print('#######################')
    return global_model, global_model.state_dict()