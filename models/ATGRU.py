# ATGRU
# Paper "Connecting targets to tweets: Semantic attention-based model for target-specific stance detection"
# http://dro.dur.ac.uk/25714/1/25714.pdf

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def Attention_Stance(hidden_unit, W_h, W_z, b_tanh, v, length, target_word):
    
    s1 = hidden_unit.size(0)
    s2 = hidden_unit.size(1)
    s3 = hidden_unit.size(2)
    
    word_tensor = torch.zeros(s1,s2,1024).cuda()
    word_tensor[:,:,:] = target_word
    
    m1 = torch.mm(hidden_unit.view(-1,hidden_unit.size(2)),W_h).view(-1, s2, s3)
    m2 = torch.mm(word_tensor.view(-1,1024),W_z).view(-1, s2, s3)
    sum_tanh = nn.functional.tanh(m1 + m2 + b_tanh.unsqueeze(0))
    u = torch.mm(sum_tanh.view(-1,s3),v.unsqueeze(1)).view(-1,s2,1).squeeze(2)
    
    for i in range(len(length)):
        u[i,length[i]:] = torch.Tensor([-1e6])
    alphas = nn.functional.softmax(u)        

    context = torch.bmm(alphas.unsqueeze(1), hidden_unit).squeeze(1)

    return context, alphas


class ATGRU(nn.Module):

    def __init__(self, linear_size, lstm_hidden_size, net_dropout, lstm_dropout):

        super(ATGRU, self).__init__()
        
        self.model_name = 'ATGRU'
        
        self.dropout = nn.Dropout(net_dropout)
        
        self.hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(1024, self.hidden_size, dropout=lstm_dropout, bidirectional=True) 
        self.linear = nn.Linear(self.hidden_size*2, linear_size)
        self.out = nn.Linear(linear_size, 3)
        self.relu = nn.ReLU()
        
        self.W_h = nn.Parameter(torch.rand([self.hidden_size*2,self.hidden_size*2],requires_grad=True))
        self.W_z = nn.Parameter(torch.rand([self.hidden_size*2,self.hidden_size*2],requires_grad=True))
        self.b_tanh = nn.Parameter(torch.rand(self.hidden_size*2,requires_grad=True))
        self.v = nn.Parameter(torch.rand(self.hidden_size*2,requires_grad=True))
        
    def forward(self, x, x_len, epoch, target_word, tokens):
        
        x = x.squeeze(1)
        target_word = target_word.squeeze(1)
        if target_word.size(1) != 1024:
            target_word = target_word.sum(1) / target_word.size(1)
        target_word = target_word.unsqueeze(1)
        
        seq_lengths, perm_idx = x_len.sort(0, descending=True)
        seq_tensor = x[perm_idx,:,:]
        packed_input = pack_padded_sequence(seq_tensor, seq_lengths,batch_first=True)
        packed_output, (ht, ct) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output,batch_first=True)
        _, unperm_idx = perm_idx.sort(0)
        h_lstm = output[unperm_idx,:,:]
        
        atten, alpha = Attention_Stance(h_lstm,self.W_h,self.W_z,self.b_tanh,self.v,x_len,epoch,tokens,target_word)
        atten = self.dropout(atten)
        
        linear = self.relu(self.linear(atten))
        linear = self.dropout(linear)
        out = self.out(linear)
        
        return out
