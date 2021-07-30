# GCAE Model
# Paper "Aspect Based Sentiment Analysis with Gated Convolutional Networks"
# https://www.aclweb.org/anthology/P18-1234.pdf

import torch
from torch import nn
from torch.nn import functional as F

def kmax_pooling(x, dim, k):

    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]

    return x.gather(dim, index)

class GCAE(nn.Module):

    def __init__(self, in_channels, out_channels, linear_size, kernel_size, kernel_size2, net_dropout):

        super(GCAE, self).__init__()
        
        self.model_name = 'GCAE_Model'

        self.dropout = nn.Dropout(net_dropout)
        
        self.k = 1
        self.linear = nn.Linear(out_channels*4, linear_size)
        self.out = nn.Linear(linear_size, 3)
        self.relu = nn.ReLU()
        self.linear5 = nn.Linear(out_channels*8*self.k, out_channels*4*self.k)
        
        self.convs_tanh = nn.ModuleList([nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=K) for K in kernel_size])
        self.convs_relu = nn.ModuleList([nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=K) for K in kernel_size])
        self.V = nn.Parameter(torch.rand([1024,out_channels],requires_grad=True).cuda())
        self.conv4 = nn.Conv2d(in_channels=in_channels,out_channels=1024,kernel_size=kernel_size2)
        
    def forward(self, x, target_word, out_channels):
        
        target_word = target_word.squeeze(1)
        if target_word.size(1) != 300:
            target_word = target_word.sum(1) / target_word.size(1)
    
        conved_tanh = [F.tanh(conv(x).squeeze(3)) for conv in self.convs_tanh]
        conved_relu = [self.relu(conv(x).squeeze(3)+torch.mm(target_word,self.V).unsqueeze(2)) for conv in self.convs_relu]
        conved_mul = [i*j for i,j in zip(conved_tanh,conved_relu)]
        pooled = [kmax_pooling(i, 2, self.k).view(-1,out_channels*self.k) for i in conved_mul]
        cat = torch.cat(pooled, dim=1) 
        
        linear = self.relu(self.linear(cat))
        linear = self.dropout(linear)
        out = self.out(linear)
        
        return out
