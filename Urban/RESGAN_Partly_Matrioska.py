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
setproctitle.setproctitle("traffic_gene@hsd")
from RESGAN_Partly import gpu
torch.manual_seed(5)
use_cuda = torch.cuda.is_available()
		

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
    hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend, output = netG(torch.cat((noise, kge), 1))#, kge)
    return hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend, output
	
from RESGAN_Partly import GeneratorP_ALL_LN_Matrioska
from RESGAN_Partly import DiscriminatorTCN
from RESGAN_Partly import Discriminator_HD
from RESGAN_Partly import Discriminator_DW
from RESGAN_Partly import save_dir_head
from RESGAN_Partly import gpu
from RESGAN_Partly import MyDataset
from RESGAN_Partly import EXP_NOISE
USE_KGE = True
NOISE_SIZE = 32
KGE_SIZE = 32
LENGTH = 672
BATCH_SIZE = 32
BATCH_FIRST = True 

dataset=MyDataset('bs_record_energy_normalized_sampled_2.npz')
DATASET_SIZE = len(dataset)
gene_size = DATASET_SIZE

real_dataset_list = [dataset.data['hours_in_weekday'], dataset.data['hours_in_weekend'], dataset.data['days_in_weekday'],\
                     dataset.data['days_in_weekend'], dataset.data['bs_record']]
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
num_layers = 2
netG = GeneratorP_ALL_LN_Matrioska()
netD_0 = Discriminator_HD(patterns = hours_in_weekday_patterns)
netD_1 = Discriminator_HD(patterns = hours_in_weekend_patterns)
netD_2 = Discriminator_DW(patterns = days_in_weekday_patterns)
netD_3 = Discriminator_DW(patterns = days_in_weekend_patterns)
netD_4 = DiscriminatorTCN()
start_it = 0

TRAIN = True
TRAIN_WM = False
TRAIN_HD = False
TRAIN_DW = False
LOAD_PRE_TRAIN = False
LAMBDA = 10
ITERS = 331
CRITIC_ITERS = 5

if use_cuda:
    netD_0 = netD_0.cuda(gpu)
    netD_1 = netD_1.cuda(gpu)
    netD_2 = netD_2.cuda(gpu)
    netD_3 = netD_3.cuda(gpu)
    netD_4 = netD_4.cuda(gpu)
    netG = netG.cuda(gpu)
netD_list = [netD_0, netD_1, netD_2, netD_3, netD_4]

sub_dir_list = ['HDD', 'HDE', 'DWD', 'DWE', 'ALL']
	

for ii in np.arange(len(netD_list)):    
    netD = netD_list[ii]
    sub_dir = sub_dir_list[ii]
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    one = torch.tensor(1, dtype=torch.float32)
    mone = one * -1.0
    if use_cuda:
        one = one.cuda(gpu)
        mone = mone.cuda(gpu)
    print(time.localtime())
    for iteration in trange(ITERS):
        start_time = time.time()
        #print(time.localtime(), ' iteration: ', iteration)
        for idx, data in enumerate(data_loader):
            if True:
                for p in netD.parameters():  
                    p.requires_grad = True  
                data_batch, kge_batch, hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend= data[1:]
                data_batch = data_batch.squeeze(1)
#                if not(BATCH_FIRST):
#                    data_batch = data_batch.permute(2,0,1)
                
                netD.zero_grad()
                
                real_data = [hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend, data_batch]

                if use_cuda:
                    real_data[ii] = real_data[ii].cuda(gpu)
                    kge_batch = kge_batch.cuda(gpu)
                D_real = netD(real_data[ii], kge_batch)
                D_real = D_real.mean()

                D_real.backward(mone)
                
                # generate noise
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)#, LENGTH)
                noise_batch = noise_batch.exponential_() if EXP_NOISE else noise_batch
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu) 

                fake_data = netG(torch.cat((noise_batch, kge_batch), 1))

                D_fake = netD(fake_data[ii], kge_batch)
                D_fake = D_fake.mean()
                D_fake.backward(one)
                
                # train with gradient penalty
                gradient_penalty = calc_gradient_penalty(netD, real_data[ii], fake_data[ii], kge_batch)
                gradient_penalty.backward()
                
                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake
                optimizerD.step()
                #print('#######Wasserstein_D###########',Wasserstein_D)
                
            if idx%CRITIC_ITERS == 0:
                for p in netD.parameters():
                    p.requires_grad = False  
                netG.zero_grad()
           
                noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)
                noise_batch = noise_batch.exponential_() if EXP_NOISE else noise_batch
                if use_cuda:
                    noise_batch = noise_batch.cuda(gpu) 
                fake = netG(torch.cat((noise_batch, kge_batch), 1))
                for fake_value in fake[0:ii]:
                    fake_value.detach_() 
                G = netD(fake[ii], kge_batch)
                G = G.mean()
                G.backward(mone)
                G_cost = -G
                optimizerG.step()
                #print(G_cost, D_cost, Wasserstein_D)
                G_costList.append(G_cost.cpu().data.numpy())
        if iteration % 30 == 0:
            save_dir = save_dir_head + '/' + sub_dir + '/iteration-' + str(iteration+start_it)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                    
            torch.save(netD.state_dict(), os.path.join(save_dir, 'netD'))
            torch.save(netG.state_dict(), os.path.join(save_dir, 'netG'))
            torch.save(netG.generator_hdd.hd_patterns, os.path.join(save_dir, 'netG.generator_hdd.hd_patterns'))
            torch.save(netG.generator_hde.hd_patterns, os.path.join(save_dir, 'netG.generator_hde.hd_patterns'))
            torch.save(netG.generator_dwd.dwr_patterns, os.path.join(save_dir, 'netG.generator_dwd.dwr_patterns'))
            torch.save(netG.generator_dwe.dwr_patterns, os.path.join(save_dir, 'netG.generator_dwe.dwr_patterns'))
            
            np.savez(os.path.join(save_dir, 'cost_generated.npz'), \
                     time=np.array(TimeList), \
                     D_cost=np.array(D_costList), \
                     sparsity=np.array(sparsityList),\
                     G_cost=np.array(G_costList),\
                     WD=np.array(WDList)) 
    
            fake_data = generate_data(netG, dataset.kge, gene_size)
            generated_data = fake_data[4].reshape(gene_size, -1).cpu().detach().numpy()
            hours_in_weekday = fake_data[0].view(gene_size, 24).cpu().detach().numpy()
            hours_in_weekend = fake_data[1].view(gene_size, 24).cpu().detach().numpy()
            days_in_weekday = fake_data[2].view(gene_size, 24*5).cpu().detach().numpy()
            days_in_weekend = fake_data[3].view(gene_size, 24*2).cpu().detach().numpy()
            fig, ax = plt.subplots(figsize=(24, 16))
            n_bins = 100
            line_w = 2
            use_cumulative = -1
            use_log = True
            n_real, bins, patches = ax.hist(real_dataset_list[ii].flatten(), n_bins, density=True, histtype='step', cumulative=use_cumulative, label='real', log=use_log, facecolor='g', linewidth=line_w)
            fake_data_now = fake_data[ii].cpu().detach().numpy()
            n_gene, bins, patches = ax.hist(fake_data_now.flatten(), n_bins, density=True, histtype='step', cumulative=use_cumulative, label='gene', log=use_log, facecolor='b', linewidth=line_w)
            ax.grid(True)
            ax.legend(loc='right')
            ax.set_title('Cumulative step histograms')
            ax.set_xlabel('Value')
            ax.set_ylabel('Likelihood of occurrence')
            plt.savefig(os.path.join(save_dir, 'fig_hist.jpg'))
            plt.close()
            dst = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)        
            dst_list.append(dst)
            np.savez(os.path.join(save_dir, 'generated.npz'), \
            generated_data = generated_data, \
            distance = dst)
            #print(G_cost, D_cost, Wasserstein_D, dst)
    print(sub_dir + ' finished!')
print('Train finished!')

