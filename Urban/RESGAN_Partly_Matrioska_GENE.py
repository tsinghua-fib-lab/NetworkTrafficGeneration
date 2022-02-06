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

gpu=4
torch.manual_seed(5)
use_cuda = torch.cuda.is_available()
from RESGAN_Partly import MyDataset
dataset=MyDataset('bs_record_energy_normalized_sampled.npz')
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
              self.softmax(self.norm_4hd(self.linear_4hd(x[:, 0:32])))
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
        self.act = nn.Sigmoid() if activation=='sigmoid' else nn.GELU() 
        self.dwr_patterns = patterns
        self.dwr_patterns.requires_grad = True

    def forward(self, x, hours_in_day):
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
        return self.act(y.squeeze(1))
	
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
		

def calc_gradient_penalty(netD, real_data, fake_data, kge):
    LAMBDA = 10
    alpha = torch.rand(BATCH_SIZE, 1)

    if use_cuda:
        alpha = alpha.cuda(gpu)
    alpha = alpha.expand(real_data.size())
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, kge)
    
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def generate_data(netG, kge, gene_size=128):
    noise = torch.randn(gene_size, NOISE_SIZE)
    noise = noise.exponential_() if EXP_NOISE else noise
    if use_cuda:
        noise = noise.cuda(gpu) 
        kge = kge.cuda(gpu)
    hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend, output = netG(torch.cat((noise, kge), 1))
    return hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend, output
	
EXP_NOISE = True
NOISE_SIZE = 32
KGE_SIZE = 32
LENGTH = 672
BATCH_SIZE = 32

GENE_NUM = 20
save_dir_head_list = ["generated_data_1002_kge_n_train_on_", "./generated_data_1012_train_on_"]
dataset_list = [0, 1, 2]

for i_USE_KGE in np.arange(1,len(save_dir_head_list)):
    USE_KGE = i_USE_KGE
    netG = GeneratorP_ALL_LN_Matrioska()
    if use_cuda:
        netG = netG.cuda(gpu)
    for i_TRAIN in np.arange(len(dataset_list)):
        pretrained_netG = torch.load(save_dir_head_list[i_USE_KGE]+str(dataset_list[i_TRAIN])+'/ALL/iteration-330/netG',map_location=torch.device(gpu))
        netG.load_state_dict(pretrained_netG)
        netG.generator_hdd.hd_patterns = torch.load(save_dir_head_list[i_USE_KGE]+str(dataset_list[i_TRAIN])+'/ALL/iteration-330/netG.generator_hdd.hd_patterns' ,map_location=torch.device(gpu))
        netG.generator_hde.hd_patterns = torch.load(save_dir_head_list[i_USE_KGE]+str(dataset_list[i_TRAIN])+'/ALL/iteration-330/netG.generator_hde.hd_patterns' ,map_location=torch.device(gpu))
        netG.generator_dwd.dwr_patterns= torch.load(save_dir_head_list[i_USE_KGE]+str(dataset_list[i_TRAIN])+'/ALL/iteration-330/netG.generator_dwd.dwr_patterns',map_location=torch.device(gpu))
        netG.generator_dwe.dwr_patterns= torch.load(save_dir_head_list[i_USE_KGE]+str(dataset_list[i_TRAIN])+'/ALL/iteration-330/netG.generator_dwe.dwr_patterns',map_location=torch.device(gpu))
        print(netG)

        for i_GENE in np.arange(len(dataset_list)):
            dataset=MyDataset('bs_record_energy_normalized_sampled_'+str(dataset_list[i_GENE])+'.npz')
            DATASET_SIZE = len(dataset)
            gene_size = DATASET_SIZE
            gene_piece_list = np.arange(0, DATASET_SIZE, gene_size).tolist()
            gene_piece_list.append(DATASET_SIZE)
            for i_GENE_NUM in np.arange(GENE_NUM):
                torch.manual_seed(int(time.time()))
                save_dir = './generated_data_all_1012_kge'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)	    
                generated_data = []
                for ii in np.arange(len(gene_piece_list)-1):
                    generated_data_temp\
                           = generate_data(netG, dataset.kge[gene_piece_list[ii]:gene_piece_list[ii+1]], gene_piece_list[ii+1]-gene_piece_list[ii])
                    generated_data.append(generated_data_temp[4].view(gene_piece_list[ii+1]-gene_piece_list[ii], -1).cpu().detach().numpy())
                generated_data = np.concatenate(generated_data)
                print(generated_data.max(), generated_data.min())
                np.savez(os.path.join(save_dir, 'RESGAN_' + str(i_USE_KGE) + '_' + str(i_TRAIN) + '_' + str(i_GENE) + '_' + str(i_GENE_NUM) + '.npz'), \
                generated_data = generated_data)
