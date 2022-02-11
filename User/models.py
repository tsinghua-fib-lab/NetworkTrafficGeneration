import os, sys
import time
import copy
import matplotlib
import random
import torch
import math
import warnings
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from tqdm import trange
import setproctitle  
setproctitle.setproctitle("traffic_gene")

USE_KGE = True


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        return x.contiguous()

class soft_exponential(nn.Module):

    def __init__(self, in_features, alpha = None):
        super(soft_exponential,self).__init__()
        self.in_features = in_features

        if alpha == None:
            self.alpha = Parameter(torch.tensor(0.0)) 
        else:
            self.alpha = Parameter(torch.tensor(alpha)) 
            
        self.alpha.requiresGrad = True 

    def forward(self, x):
            return x

        if (self.alpha < 0.0):
            return - torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        if (self.alpha > 0.0):
            return (torch.exp(self.alpha * x) - 1)/ self.alpha + self.alpha
            
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, padding_mode='circular'):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation))
        self.chomp1 = Chomp1d(padding/2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation))
        self.chomp2 = Chomp1d(padding/2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        index_2 = int((out.shape[2]-x.shape[2])/2)
        out = out[:,:,index_2:index_2+x.shape[2]]
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dilation_size_list=''):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            if dilation_size_list:
                dilation_size = dilation_size_list[i]
            else:
                dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            dilation_size = int(288/kernel_size) if kernel_size*dilation_size>288 else dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,\
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class UserTrafficDataset(Dataset):
    def __init__(self, data='/data/utf_dataset_file.npz', \
                 file_num=6055, dense=True, cluster=None):
        self.data = np.load(data)
        try:
            if dense:
                self.dense_tag = self.data['dense_tag']
                self.user_id = self.data['user_id'][self.dense_tag][0:file_num]
                self.app_usage = torch.from_numpy(self.data['app_usage'][self.dense_tag][0:file_num].astype(np.float32))
                self.geo_sta = torch.from_numpy(self.data['geo_sta'][self.dense_tag][0:file_num].astype(np.float32))
                self.utf = torch.from_numpy(self.data['utf'][self.dense_tag][0:file_num].astype(np.float32))
                self.utf_norm = torch.from_numpy(self.data['utf_norm'][self.dense_tag][0:file_num].astype(np.float32))
                if cluster == 'kmeans':
                    self.cluster_label = torch.tensor(self.data['Kclusters4dense'], dtype=torch.int64)
                elif cluster == 'spectral':
                    self.cluster_label = torch.tensor(self.data['Sclusters4dense'], dtype=torch.int64)
                else:
                    self.cluster_label = -1. * torch.zeros(len(self.user_id))
            else:
                self.user_id = self.data['user_id'][0:file_num]
                self.app_usage = torch.from_numpy(self.data['app_usage'][0:file_num].astype(np.float32))
                self.geo_sta = torch.from_numpy(self.data['geo_sta'][0:file_num].astype(np.float32))
                self.utf = torch.from_numpy(self.data['utf'][0:file_num].astype(np.float32))
                self.utf_norm = torch.from_numpy(self.data['utf_norm'][0:file_num].astype(np.float32))
                self.cluster_label =  -1. * torch.zeros(len(self.user_id))

        except Exception as E:
            print(E)
            print('load data error!')

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        user_id = self.user_id[idx]
        app_usage = self.app_usage[idx]
        geo_sta = self.geo_sta[idx]
        utf_norm = self.utf_norm[idx]
        utf = self.utf[idx]
        cluster_label = self.cluster_label[idx]
        return user_id, app_usage, geo_sta, utf_norm, utf, cluster_label
    def __len__(self):
        return len(self.user_id)


class TCNDiscriminator(nn.Module):
    def __init__(self, input_size=1, num_clusters=6, num_channels=None, condition_size=32, condition_squeeze_size=4, dropout=0.3):
        super(TCNDiscriminator, self).__init__()
        if num_channels is None:
            num_channels = [1, 2, 4, 8, num_clusters, num_clusters]
        self.tcn_d = TemporalConvNet(input_size, num_channels, kernel_size=48, dropout=dropout)
        self.tcn_w = TemporalConvNet(input_size, num_channels, kernel_size=48*6, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1]*2+condition_squeeze_size, num_clusters+1) if USE_KGE else \
                      nn.Linear(num_channels[-1]*2, num_clusters+1)
        self.linear_kge = nn.Linear(condition_size, condition_squeeze_size)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.linear_kge.weight.data.normal_(0, 0.01)

    def forward(self, x, kge):
        x = x.unsqueeze(1)
        y_d = self.tcn_d(x)[:,:,-1]
        y_w = self.tcn_w(x)[:,:,-1]
        y = self.linear(torch.cat((y_d, y_w, self.linear_kge(kge)), 1)) if USE_KGE else \
            self.linear(torch.cat((y_d, y_w), 1))
        if self.linear.weight.shape[0] >= self.linear.weight.shape[1]:
            weight_mm = torch.mm(self.linear.weight.transpose(1, 0), self.linear.weight)
        else:
            weight_mm = torch.mm(self.linear.weight, self.linear.weight.transpose(1, 0))
        return self.softmax(y), weight_mm



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0., max_len).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model)) 

        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)  

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x)#(x.permute(1,0,2)) # size = [L, batch, d_model]

class BiLSTMGenerator(nn.Module):
    def __init__(self, input_size=None, noise_size=32, out_seq_size=288, hidden_size=24, condition_size=32, dropout=0.3,
                 num_layers=3, bidirectional=True, head_num=1):
        super(BiLSTMGenerator, self).__init__()
        if input_size is None:
            input_size = hidden_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                           dropout=dropout, bidirectional=bidirectional)

        self.linear_c0 = nn.Linear(condition_size, hidden_size * num_layers * 2)
        self.norm_c0 = nn.LayerNorm(hidden_size * num_layers * 2)

        self.linear_h0 = nn.Linear(noise_size, hidden_size * num_layers * 2)
        self.norm_h0 = nn.LayerNorm(hidden_size * num_layers * 2)

        self.linear_x = nn.Linear(noise_size + condition_size, out_seq_size) if USE_KGE else nn.Linear(noise_size,
                                                                                                       out_seq_size)
        self.norm_x = nn.LayerNorm(out_seq_size)
        self.pe = PositionalEncoding(1, dropout)

        self.query_linear = nn.Linear(hidden_size*(int(bidirectional)+1), hidden_size*(int(bidirectional)+1))
        self.key_linear = nn.Linear(hidden_size*(int(bidirectional)+1), hidden_size*(int(bidirectional)+1))
        self.value_linear = nn.Linear(hidden_size*(int(bidirectional)+1), hidden_size*(int(bidirectional)+1))
        self.attn = nn.MultiheadAttention(hidden_size*(int(bidirectional)+1), num_heads=head_num, batch_first=True)

        self.linear_out = nn.Linear(hidden_size*(int(bidirectional)+1), input_size)
        self.norm_out = nn.LayerNorm(input_size)

        self.hidden_size = hidden_size
        self.out_seq_size = out_seq_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional

        self.relu = nn.ReLU()

    def forward(self, x, kge):
        batch_size = x.shape[0]
        c0 = self.norm_c0(self.linear_c0(kge)).reshape(batch_size, self.num_layers * 2, self.hidden_size).permute(1, 0, 2).contiguous() \
            if USE_KGE else \
            self.norm_c0(self.linear_c0(torch.zeros_like(kge))).reshape(
                batch_size, self.num_layers * 2, self.hidden_size).permute(1, 0,2).contiguous()
        h0 = self.norm_h0(self.linear_h0(x)).reshape(batch_size, self.num_layers * 2, self.hidden_size).permute(1, 0, 2).contiguous()
        x = torch.cat((x, kge), 1) if USE_KGE else x
        x = self.norm_x(self.linear_x(x)).unsqueeze(2)
        x = self.pe(x).reshape(batch_size, int(self.out_seq_size/self.input_size), self.input_size)
        x, (hn, cn) = self.rnn(x, (h0, c0))
        query_proj = self.query_linear(x)
        key_proj = self.key_linear(x)
        value_proj = self.value_linear(x)
        x, attn_output_weights = self.attn(query_proj, key_proj, value_proj)
        x = self.norm_out(self.linear_out(x)).reshape(batch_size, self.out_seq_size)
        x = self.relu(x)
        return x

class PatternGenerator(nn.Module):
    def __init__(self, pattern_num=6, input_size=None, noise_size=32, out_seq_size=288, hidden_size=24, condition_size=32, dropout=0.3,
                 num_layers=3, bidirectional=True, head_num=1):
        super(PatternGenerator, self).__init__()
        self.pattern_num = pattern_num
        self.generators = nn.ModuleList([])
        for ii in range(pattern_num):
            self.generators.append(BiLSTMGenerator(input_size=input_size,
                                                   noise_size=noise_size,
                                                   out_seq_size=out_seq_size,
                                                   hidden_size=hidden_size,
                                                   condition_size=condition_size,
                                                   dropout=dropout,
                                                   num_layers=num_layers,
                                                   bidirectional=bidirectional,
                                                   head_num=head_num)
                                   )
    def forward(self, x, kge):
        x = torch.cat(
            [self.generators[ii](x, kge).unsqueeze(1) for ii in range(self.pattern_num)], 1
        )
        return x

class MarkovSwitchGenerator(nn.Module):
    def __init__(self, pattern_num=6, noise_size=32, out_seq_size=288, condition_size=32, tau=0.1):
        super(MarkovSwitchGenerator, self).__init__()
        self.linear_x0 = nn.Linear(condition_size+noise_size, pattern_num) if USE_KGE else nn.Linear(noise_size, pattern_num)
        self.norm_x0 = nn.LayerNorm(pattern_num)

        self.linear = nn.Linear(pattern_num, pattern_num)
        self.norm = nn.LayerNorm(pattern_num)

        self.relu = nn.ReLU()

        self.out_seq_size = out_seq_size
        self.pattern_num = pattern_num
        self.tau = tau

        self.init_weights()

    def init_weights(self):
        self.linear.weight = torch.nn.Parameter(torch.eye(self.pattern_num)+torch.rand(self.pattern_num,self.pattern_num)/10)

    def forward(self, x, kge):
        batch_size = x.shape[0]
        x = torch.cat((x, kge), 1) if USE_KGE else x
        p_0 = self.linear_x0(self.relu(x))
        x = F.gumbel_softmax(p_0, tau=0.1, hard=True, eps=1e-10, dim=1)
        out = x.unsqueeze(2)

        for t in range(self.out_seq_size-1):
            p_t = self.relu(self.norm(self.linear(x)))
            x = F.gumbel_softmax(p_t, tau=self.tau, hard=True, eps=1e-10, dim=1)
            out = torch.cat((out, x.unsqueeze(2)), 2)
        return out
