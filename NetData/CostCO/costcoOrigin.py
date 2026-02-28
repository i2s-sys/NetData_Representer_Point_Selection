import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch as t

if t.cuda.is_available():
    device = 'cuda:0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = t.device('cpu')

device=torch.device('cuda:0' if torch.cuda.is_available() else 'CPU')


class CostCo(nn.Module):
    #from kdd 18
    #paper:CoSTCo: A Neural Tensor Completion Model for Sparse Tensors
    def __init__(self,i_size, j_size, k_size, embedding_dim,nc=100):
        super(CostCo,self).__init__()
        self.iembeddings = nn.Embedding(i_size, embedding_dim)
        self.jembeddings = nn.Embedding(j_size, embedding_dim)
        self.kembeddings = nn.Embedding(k_size, embedding_dim)
        self.conv1=nn.Conv2d(1,nc,(1,embedding_dim))
        self.conv2=nn.Conv2d(nc,nc,(3,1))
        self.fc1=nn.Linear(nc,1)


    def forward(self,i_input, j_input, k_input):

        # embedding
        lookup_itensor = i_input.long()
        lookup_jtensor = j_input.long()
        lookup_ktensor = k_input.long()
        i_embeds = self.iembeddings(lookup_itensor).unsqueeze(1)
        j_embeds = self.jembeddings(lookup_jtensor).unsqueeze(1)
        k_embeds = self.kembeddings(lookup_ktensor).unsqueeze(1)
        # i_embeds= torch.squeeze(i_embeds)
        # j_embeds= torch.squeeze(j_embeds)
        # k_embeds= torch.squeeze(k_embeds)

        H=t.cat((i_embeds,j_embeds,k_embeds),2)
        # H=H.unsqueeze(1)

        x=t.relu(self.conv1(H))
        x=t.relu(self.conv2(x))
        x=x.view(-1,x.shape[1])
        x=self.fc1(x)

        return x