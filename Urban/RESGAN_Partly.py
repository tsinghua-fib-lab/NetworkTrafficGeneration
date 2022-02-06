import os, sys
import time
import copy
import matplotlib
import random
matplotlib.use('SVG')
import torch
import math
import warnings
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from tqdm import trange
import setproctitle  
setproctitle.setproctitle("traffic_gene")
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = np.load(data)
        try:
            self.bs_id = self.data['bs_id_kge']
            self.bs_record = torch.from_numpy(self.data['bs_record_kge'].astype(np.float32)).reshape(self.bs_id.shape[0],1,LENGTH)
            self.kge = torch.from_numpy(self.data['bs_kge'].astype(np.float32))/40.0
            self.hours_in_weekday = torch.from_numpy(self.data['hours_in_weekday'].astype(np.float32))
            self.hours_in_weekend = torch.from_numpy(self.data['hours_in_weekend'].astype(np.float32))
            self.days_in_weekday = torch.from_numpy(self.data['days_in_weekday'].astype(np.float32))
            self.days_in_weekend = torch.from_numpy(self.data['days_in_weekend'].astype(np.float32))
            self.days_in_weekday_residual = torch.from_numpy(self.data['days_in_weekday_residual'].astype(np.float32))
            self.days_in_weekend_residual = torch.from_numpy(self.data['days_in_weekend_residual'].astype(np.float32))
            self.weeks_in_month_residual = torch.from_numpy(self.data['weeks_in_month_residual'].astype(np.float32))
            self.hours_in_weekday_patterns = torch.from_numpy(self.data['hours_in_weekday_patterns'].astype(np.float32))
            self.hours_in_weekend_patterns = torch.from_numpy(self.data['hours_in_weekend_patterns'].astype(np.float32))
            self.days_in_weekday_patterns = torch.from_numpy(self.data['days_in_weekday_patterns'].astype(np.float32))
            self.days_in_weekend_patterns = torch.from_numpy(self.data['days_in_weekend_patterns'].astype(np.float32))
            self.days_in_weekday_residual_patterns = torch.from_numpy(self.data['days_in_weekday_residual_patterns'].astype(np.float32))
            self.days_in_weekend_residual_patterns = torch.from_numpy(self.data['days_in_weekend_residual_patterns'].astype(np.float32))

        except:
            self.bs_id = self.data['bs_id']
            self.bs_record = torch.from_numpy(self.data['bs_record'].astype(np.float32)).reshape(self.bs_id.shape[0],1,LENGTH)
            self.kge = torch.from_numpy(self.data['bs_kge'].astype(np.float32))/40.0
            self.hours_in_weekday = torch.from_numpy(self.data['hours_in_weekday'].astype(np.float32))
            self.hours_in_weekend = torch.from_numpy(self.data['hours_in_weekend'].astype(np.float32))
            self.days_in_weekday = torch.from_numpy(self.data['days_in_weekday'].astype(np.float32))
            self.days_in_weekend = torch.from_numpy(self.data['days_in_weekend'].astype(np.float32))
            self.days_in_weekday_residual = torch.from_numpy(self.data['days_in_weekday_residual'].astype(np.float32))
            self.days_in_weekend_residual = torch.from_numpy(self.data['days_in_weekend_residual'].astype(np.float32))
            self.weeks_in_month_residual = torch.from_numpy(self.data['weeks_in_month_residual'].astype(np.float32))
            self.hours_in_weekday_patterns = torch.from_numpy(self.data['hours_in_weekday_patterns'].astype(np.float32))
            self.hours_in_weekend_patterns = torch.from_numpy(self.data['hours_in_weekend_patterns'].astype(np.float32))
            self.days_in_weekday_patterns = torch.from_numpy(self.data['days_in_weekday_patterns'].astype(np.float32))
            self.days_in_weekend_patterns = torch.from_numpy(self.data['days_in_weekend_patterns'].astype(np.float32))
            self.days_in_weekday_residual_patterns = torch.from_numpy(self.data['days_in_weekday_residual_patterns'].astype(np.float32))
            self.days_in_weekend_residual_patterns = torch.from_numpy(self.data['days_in_weekend_residual_patterns'].astype(np.float32))
			
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        bs_id = self.bs_id[idx]
        bs_record = self.bs_record[idx]
        kge = self.kge[idx]
        hours_in_weekday = self.hours_in_weekday[idx]
        hours_in_weekend = self.hours_in_weekend[idx]
        days_in_weekday = self.days_in_weekday[idx]
        days_in_weekend = self.days_in_weekend[idx]
        return bs_id, bs_record, kge, hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend
    def __len__(self):
        return self.bs_id.shape[0] 

gpu=3
torch.manual_seed(5)
use_cuda = torch.cuda.is_available()

NOISE_SIZE = 32
KGE_SIZE = 32
SHAPE_M0 = 4
SHAPE = [ (4, 7*24*SHAPE_M0), (4*7, 24*SHAPE_M0), (4*7*24, SHAPE_M0) ]
LENGTH = 672
BATCH_SIZE = 256
BATCH_FIRST = False 
save_dir_head = "./generated_data_1012_train_on_2"
EXP_NOISE = True
ACT = True
USE_KGE = True#False
TimeList = []
D_costList = []
G_costList = []
sparsityList = []
WDList = []
dst_list = []
dataset=MyDataset('bs_record_energy_normalized_sampled.npz')
gene_size = 1024
DATASET_SIZE = len(dataset)
hours_in_weekday_patterns = dataset.hours_in_weekday_patterns
hours_in_weekend_patterns = dataset.hours_in_weekend_patterns
days_in_weekday_patterns = dataset.days_in_weekday_patterns
days_in_weekend_patterns = dataset.days_in_weekend_patterns
days_in_weekday_residual_patterns = dataset.days_in_weekday_residual_patterns
days_in_weekend_residual_patterns = dataset.days_in_weekend_residual_patterns
if use_cuda:
    hours_in_weekday_patterns = hours_in_weekday_patterns.cuda(gpu)
    hours_in_weekend_patterns = hours_in_weekend_patterns.cuda(gpu)
    days_in_weekday_patterns = days_in_weekday_patterns.cuda(gpu)
    days_in_weekend_patterns = days_in_weekend_patterns.cuda(gpu)
    days_in_weekday_residual_patterns = days_in_weekday_residual_patterns.cuda(gpu)
    days_in_weekend_residual_patterns = days_in_weekend_residual_patterns.cuda(gpu)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

dropout = 0.3
num_layers = 6
nhead = [4, 2, 1]
LAMBDA = 10


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
        if (self.alpha == 0.0):
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
            dilation_size = int(672/kernel_size) if kernel_size*dilation_size>672 else dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        #print('xx', x.shape)
        return self.network(x)

class MLPNet(nn.Module):
    def __init__(self, dim_list, dropout=0.5):
        super(MLPNet, self).__init__()
        layers = []
        num_layers = len(dim_list) - 1
        for i in range(num_layers):
            layers += [nn.Linear(dim_list[i], dim_list[i+1])]
        layers += [nn.ReLU()]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)




class GeneratorP_HD_SUM(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="relu", patterns = hours_in_weekday_patterns):
        super(GeneratorP_HD_SUM, self).__init__()
        self.linear_4hd =  nn.Linear(NOISE_SIZE+KGE_SIZE, pattern_num) if USE_KGE else  \
                           nn.Linear(NOISE_SIZE, pattern_num)
        self.norm_4hd = nn.LayerNorm(pattern_num)
        self.softmax = nn.Softmax(dim=1)
#        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(patterns.shape[1])
        self.act = nn.Sigmoid() if activation=='sigmoid' else nn.GELU() 
        self.hd_patterns = patterns

    def forward(self, x):
        x_p = self.softmax(self.norm_4hd(self.linear_4hd(x))) if USE_KGE else  \
              self.softmax(self.norm_4hd(self.linear_4hd(x[:, 0:32])))#[:, 32:])))
        hours_in_day = self.act(self.norm(x_p@self.hd_patterns))
        return hours_in_day
		


class GeneratorP_DW_SUM(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="relu", patterns = days_in_weekday_residual_patterns):
        super(GeneratorP_DW_SUM, self).__init__()
        self.linear_4dw =  nn.Linear(NOISE_SIZE+KGE_SIZE, pattern_num) if USE_KGE else  \
                           nn.Linear(NOISE_SIZE, pattern_num)		
        self.norm_4dw = nn.LayerNorm(pattern_num)
        self.norm = nn.LayerNorm(patterns.shape[1])
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
#        self.relu = nn.ReLU()
        self.act = nn.Sigmoid() if activation=='sigmoid' else nn.GELU() 
        self.dwr_patterns = patterns
        self.dwr_patterns.requires_grad = True

    def forward(self, x, hours_in_day):
#        BZ = x.shape[0]
        days_in_week_residual = self.softmax(self.norm_4dw(self.linear_4dw(x))) if USE_KGE else  \
                                self.softmax(self.norm_4dw(self.linear_4dw(x[:, 0:32])))
        days_in_week_residual = self.norm(days_in_week_residual@self.dwr_patterns)
        days_in_week = days_in_week_residual + hours_in_day.repeat(1, int(days_in_week_residual.shape[1]/hours_in_day.shape[1]))
        return self.act(days_in_week)


class GeneratorP_WM_SUM(nn.Module):
    def __init__(self, input_size=20, num_channels=[1]*6, kernel_size=[24, 7*24, 28*24], dropout=0.3, kge_size=32, kge_squeeze_size=10, activation="relu"):
        super(GeneratorP_WM_SUM, self).__init__()
        self.linear_kge = nn.Linear(kge_size, kge_squeeze_size)
        self.norm_kge = nn.LayerNorm(kge_squeeze_size)
        input_size = 20 if USE_KGE else 10
        self.tcn_d = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[0], dropout=dropout)
        self.tcn_w = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[1], dropout=dropout)
        self.tcn_m = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[2], dropout=dropout)
        self.linear_4wm = nn.Linear(32, kge_squeeze_size*672)
        self.norm_4wm = nn.LayerNorm(kge_squeeze_size*672)
        self.norm = nn.LayerNorm([1,672])
        self.act = nn.Sigmoid() if activation=='sigmoid' else nn.GELU() 

        self.linear = nn.Linear(num_channels[-1]*len(kernel_size), num_channels[-1])
        self.init_weights()
        self.tanh = nn.Tanh()
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, days_in_week):
        BZ = x.shape[0]
        kge = self.norm_kge(self.linear_kge(x[:, 32:]))
        x = self.norm_4wm(self.linear_4wm(x[:,0:32])).reshape(BZ, -1, 672)
        x = torch.cat((x, kge.view(kge.size(0),kge.size(1),1).expand(-1, -1, x.size(2))), 1) if USE_KGE else x
        y_d = self.tcn_d(x)
        y_w = self.tcn_w(x)
        y_m = self.tcn_m(x)
        y = self.norm(self.linear(torch.cat((y_d, y_w, y_m), 1).transpose(1, 2)).transpose(1, 2))
        y = y + days_in_week.repeat(1, 4).reshape(BZ,1,-1)
        return self.act(y.squeeze(1))#, y_d, y_w, y_m, kge


class GeneratorP_ALL_LN_Matrioska(nn.Module):#
    def __init__(self, nhead=[1,1,1], dim_feedforward=2048, dropout=0.3, activation="relu", num_layers=6):
        super(GeneratorP_ALL_LN_Matrioska, self).__init__()
        self.generator_hdd =  GeneratorP_HD_SUM(patterns = hours_in_weekday_patterns, activation=activation)
        self.generator_hde =  GeneratorP_HD_SUM(patterns = hours_in_weekend_patterns, activation=activation)
        self.generator_dwd =  GeneratorP_DW_SUM(patterns = days_in_weekday_residual_patterns, activation=activation)
        self.generator_dwe =  GeneratorP_DW_SUM(patterns = days_in_weekend_residual_patterns, activation=activation)
        self.generator_wm =  GeneratorP_WM_SUM(activation=activation)

    def forward(self, x):
        BZ = x.shape[0]
        hours_in_weekday = self.generator_hdd(x)
        hours_in_weekend = self.generator_hde(x)
        days_in_weekday = self.generator_dwd(x, hours_in_weekday)
        days_in_weekend = self.generator_dwe(x, hours_in_weekend)
        days_in_week = torch.cat((days_in_weekday, days_in_weekend), 1)
        tfc = self.generator_wm(x, days_in_week)
        return hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend, tfc

class Discriminator_HD(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="sigmoid", hidden_size=8, patterns=hours_in_weekday_patterns):
        super(Discriminator_HD, self).__init__()
        self.linear0 =  nn.Linear(pattern_num+KGE_SIZE, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.Sigmoid() if activation=='sigmoid' else nn.GELU() 
        self.linear1 =  nn.Linear(hidden_size, 1)	
        self.hd_patterns = patterns
    def forward(self, x, kge):        
        x_p = x@self.hd_patterns.T
        x = self.act(self.linear1(self.norm(self.linear0(torch.cat((x_p, kge), 1)))))      
        return x

class Discriminator_DW(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="sigmoid", hidden_size=8, patterns = days_in_weekday_patterns):
        super(Discriminator_DW, self).__init__()
        self.linear0 =  nn.Linear(pattern_num+KGE_SIZE, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.Sigmoid() if activation=='sigmoid' else nn.GELU() 
        self.linear1 =  nn.Linear(hidden_size, 1)	
        self.dwr_patterns = patterns
    def forward(self, x, kge):        
        x_p = x@self.dwr_patterns.T
        x = self.act(self.linear1(self.norm(self.linear0(torch.cat((x_p, kge), 1)))))        
        return x


class DiscriminatorTCN(nn.Module):
    def __init__(self, input_size=1, num_channels=[1]*6, kernel_size=[24, 7*24, 28*24], dropout=0.3, kge_size=32, kge_squeeze_size=10, activation="sigmoid"):
        super(DiscriminatorTCN, self).__init__()
        self.tcn_d = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[0], dropout=dropout)
        self.tcn_w = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[1], dropout=dropout)
        self.tcn_m = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[2], dropout=dropout)
        self.linear = nn.Linear(num_channels[-1]*len(kernel_size)+2+kge_squeeze_size, 1)
        self.linear_kge = nn.Linear(kge_size, kge_squeeze_size)
        self.init_weights()
        self.act = nn.Sigmoid() if activation=='sigmoid' else nn.GELU() 
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, kge):
        x = x.unsqueeze(1)
        x_mean = x.mean(2)
        x_min = x.min(2).values
        y_d = self.tcn_d(x)[:,:,-1]
        y_w = self.tcn_w(x)[:,:,-1]
        y_m = self.tcn_m(x)[:,:,-1]
        y = self.linear(torch.cat((y_d, y_w, y_m, x_mean, x_min, self.linear_kge(kge)), 1))
        return self.act(y)
		

def calc_gradient_penalty(netD, real_data, fake_data, kge):
    LAMBDA = 10
    alpha = torch.rand(BATCH_SIZE, 1)

    if use_cuda:
        alpha = alpha.cuda(gpu)
    alpha = alpha.expand(real_data.size())
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, kge)
    try:
        disc_interpolates = disc_interpolates[0]
    except:
        pass;

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def generate_data(netG, kge, gene_size=128):
    noise = torch.randn(gene_size, NOISE_SIZE)
    if use_cuda:
        noise = noise.cuda(gpu) 
        kge = kge.cuda(gpu)
    output, hours_in_day, days_in_week, weeks_in_month_residual = netG(torch.cat((noise, kge), 1))
    return output, hours_in_day, days_in_week, weeks_in_month_residual
	
