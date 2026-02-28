#!/usr/bin/env python
# coding: utf-8
import time
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
import pickle
import os
import torch
import torch.nn as nn

# This script assumes CUDA is available and uses GPU tensors by default.
dtype = torch.cuda.FloatTensor
import time


class softmax(nn.Module):
    def __init__(self, W):
        super(softmax, self).__init__()
        #W is a numpy array loaded from disk, converted to a trainable tensor. W是从磁盘加载的numpy数组，转换为可训练张量。
        self.W = Variable(torch.from_numpy(W).type(dtype), requires_grad=True)

    def forward(self, x, y):
        # Compute softmax cross-entropy (Phi) and L2 regularizer (L2). 计算softmax交叉熵（Phi）和L2正则化器（L2）。
        D = (torch.matmul(x,self.W))
        D_max,_ = torch.max(D,dim = 1, keepdim = True)
        D = D-D_max
        A = torch.log(torch.sum(torch.exp(D),dim = 1))
        B = torch.sum(D*y,dim=1)
        Phi = torch.sum(A-B)
        W1 = torch.squeeze(self.W)
        L2 = torch.sum(torch.mul(W1, W1))
        return (Phi,L2)

def softmax_np(x):
    # Numpy softmax for precomputed logits. 用于预计算logits。
    e_x = np.exp(x - np.max(x,axis = 1,keepdims = True))
    return e_x / e_x.sum(axis = 1,keepdims = True)

def load_data(dataset):
    if dataset == "Cifar":
        # CIFAR precomputed features/weights expected under data/. 预先计算了数据下预期的特征权重。
        with open("data/weight_323436.pkl", "rb") as input_file:
            # for python 2 run
            # [W_32,W_34,W_36,intermediate_output_32,intermediate_output_34,intermediate_output_36] = pickle.load(input_file)
            # for python 3
            [W_32,W_34,W_36,intermediate_output_32,intermediate_output_34,intermediate_output_36] = pickle.load(input_file, encoding = 'latin1')
            print((softmax_np(np.matmul(np.concatenate([intermediate_output_34,np.ones((intermediate_output_34.shape[0],1))],axis = 1),W_36))-intermediate_output_36)[:5,:])
            print(intermediate_output_36[:5,:])
        print('done loading')
        model = softmax(W_36)
        model.cuda()
        start = time.time()
        return (np.concatenate([intermediate_output_34,np.ones((intermediate_output_34.shape[0],1))],axis = 1), intermediate_output_36, model)
    elif dataset == "AwA":
        # AwA precomputed features/weights expected under data/. 数据下预期的预先计算的特征权重。
        with open("data/weight_bias.pickle", "rb") as input_file:
            # for python 2 run
            # [weight,bias] = pickle.load(input_file)
            # for python 3
            [weight,bias] = pickle.load(input_file, encoding = 'latin1')
        train_feature = np.squeeze(np.load('data/train_feature_awa.npy'))
        train_output = np.squeeze(np.load('data/train_output_awa.npy'))
        weight = np.transpose(np.concatenate([weight,np.expand_dims(bias,1)],axis = 1))
        train_feature = np.concatenate([train_feature,np.ones((train_feature.shape[0],1))],axis = 1)
        train_output = softmax_np(train_output)
        model = softmax(weight)
        model.cuda()
        return (train_feature,train_output,model)

def to_np(x):
    # Convenience: move tensor to CPU and convert to numpy.
    return x.data.cpu().numpy()

# Implementation of backtracking line search. 实现回溯线搜索。
def backtracking_line_search(optimizer,model,grad,x,y,val,beta,N,args):
    t = 10.0
    beta = 0.5
    W_O = to_np(model.W)
    grad_np = to_np(grad)
    while(True):
        model.W = Variable(torch.from_numpy(W_O-t*grad_np).type(dtype), requires_grad=True)
        val_n = 0.0
        (Phi,L2) = model(x,y)
        val_n = Phi/N + L2*args.lmbd
        if t < 0.0000000001 :
            print("t too small")
            break
        if to_np(val_n - val + t*torch.norm(grad)**2/2)>=0:
            t = beta *t
        else:
            break

# Numerically stable softmax in torch. 数值稳定的softmax
def softmax_torch(temp,N):
    max_value,_ = torch.max(temp,1,keepdim = True)
    temp = temp-max_value
    D_exp = torch.exp(temp)
    D_exp_sum = torch.sum(D_exp, dim=1).view(N,1)
    return D_exp.div(D_exp_sum.expand_as(D_exp))

def train(X, Y, model, args):
    # Optimize W, then compute representer weights via decomposition. 优化W，然后通过分解计算表征权值。
    x = Variable(torch.FloatTensor(X).cuda())
    y = Variable(torch.FloatTensor(Y).cuda())
    N = len(Y)
    min_loss = 10000.0
    optimizer = optim.SGD([model.W],lr = 1.0)
    for epoch in range(args.epoch):
        sum_loss = 0
        phi_loss = 0
        optimizer.zero_grad()
        (Phi,L2) = model(x,y)
        loss = L2*args.lmbd + Phi/N
        phi_loss += to_np(Phi/N)
        loss.backward()
        temp_W = model.W.data
        grad_loss = to_np(torch.mean(torch.abs(model.W.grad)))
        # save the W with lowest loss
        if grad_loss < min_loss:
            if epoch ==0:
                init_grad = grad_loss
            min_loss = grad_loss
            best_W = temp_W
            if min_loss < init_grad/200:
                print('stopping criteria reached in epoch :{}'.format(epoch))
                break
        backtracking_line_search(optimizer,model,model.W.grad,x,y,loss,0.5,N,args)
        if epoch % 100 == 0:
            print('Epoch:{:4d}\tloss:{}\tphi_loss:{}\tgrad:{}'.format(epoch, to_np(loss), phi_loss, grad_loss))

    # Calculate representer weights based on the theorem decomposition. 根据定理分解计算表征权值。
    temp = torch.matmul(x,Variable(best_W))
    softmax_value = softmax_torch(temp,N)
    # derivative of softmax cross entropy softmax交叉熵的导数
    weight_matrix = softmax_value-y
    weight_matrix = torch.div(weight_matrix,(-2.0*args.lmbd*N))
    print(weight_matrix[:5,:5].cpu())
    w = torch.matmul(torch.t(x),weight_matrix)
    print(w[:5,:5].cpu())
    # Predict using the decomposed representer weights. 使用分解的表示权重进行预测。
    temp = torch.matmul(x,w.cuda())
    print(temp[:5,:5].cpu())
    softmax_value = softmax_torch(temp,N)
    y_p = to_np(softmax_value)
    print(y_p[:5,:])

    print('L1 difference between ground truth prediction and prediction by representer theorem decomposition')
    print(np.mean(np.abs(to_np(y)-y_p)))

    from scipy.stats.stats import pearsonr
    print('pearson correlation between ground truth  prediction and prediciton by representer theorem')
    y = to_np(y)
    corr,_ = (pearsonr(y.flatten(),(y_p).flatten()))
    print(corr)
    sys.stdout.flush()
    return to_np(weight_matrix)

def main(args):
    # Load precomputed data/features and run representer computation. 加载预先计算的数据特征并运行表示计算。
    x,y,model = load_data(args.dataset)
    start = time.time()
    weight_matrix = train(x,y,model,args)
    end = time.time()
    print('computational time')
    print(end-start)
    # Persist representer weights for downstream analysis/visualization. 持久表示下游分析可视化的权重。
    np.savez("output/weight_matrix_{}".format(args.dataset),weight_matrix = weight_matrix)
    with open("output/weight_matrix_{}.pkl".format(args.dataset), "wb") as output_file:
        pickle.dump([weight_matrix,y], output_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    # Entry point: dataset selection and hyperparameters. 切入点：数据集选择和超参数。
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmbd', type=float, default=0.003)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--dataset', type=str, default="Cifar")
    args = parser.parse_args()
    print(args)
    main(args)
