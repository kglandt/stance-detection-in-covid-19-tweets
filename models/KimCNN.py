# Kim CNN Model

import torch
from torch import nn

def kmax_pooling(x, dim, k):

    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]

    return x.gather(dim, index)

class KimCNN(nn.Module):

    def __init__(self, in_channels, out_channels, linear_size, kernel_size, net_dropout):

        super(KimCNN, self).__init__()
        
        self.model_name = 'Kim_CNN'
        
        self.dropout = nn.Dropout(net_dropout)
        
        self.k = 1
        self.linear = nn.Linear(out_channels*4, linear_size)
        self.out = nn.Linear(128, 3)
        self.relu = nn.ReLU()
        
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=K) for K in kernel_size])
        
    def forward(self, x, out_channels):
        
        conved = [self.relu(conv(x).squeeze(3)) for conv in self.convs]
        pooled = [kmax_pooling(i, 2, self.k).view(-1,out_channels*self.k) for i in conved]
        cat = torch.cat(pooled, dim=1) 
        
        linear = self.relu(self.linear(cat))
        linear = self.dropout(linear)
        out = self.out(linear)
        
        return out
