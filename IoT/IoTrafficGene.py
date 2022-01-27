import os, sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
import torch.optim as optim
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt
import setproctitle  
setproctitle.setproctitle("IoT_traffic_gene") 

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 2

class TrafficDataset(Dataset):
    def __init__(self, data):
        self.data = np.load(data)
        self.feature = torch.split(torch.from_numpy(self.data['data_feature'].astype(np.float32)), [int(LENGTH_MAX),3100-int(LENGTH_MAX)], dim=1)[0]
        self.category = torch.from_numpy(self.data['data_category'].astype(np.float32))
        self.lengths = np.sum(self.data['data_gen_flag'], axis=1).astype(np.float32)
        self.kge = torch.from_numpy(self.data['data_embedding'].astype(np.float32))
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        feature = self.feature[idx]
        category = self.category[idx]
        lengths = self.lengths[idx]
        kge = self.kge[idx]
        return feature, category, lengths, kge
    def __len__(self):
        return self.category.shape[0] 


class CategoryGenerator(nn.Module):
    def __init__(self, category_size_o, category_size, hidden_size_etm=256):
        super().__init__()
        self.category_linear_0 = nn.Sequential(nn.Linear(category_size, hidden_size_etm), \
                                               nn.BatchNorm1d(hidden_size_etm), nn.ReLU())
        self.category_norm = nn.BatchNorm1d(hidden_size_etm)
        self.category_linear_1 = nn.Sequential(nn.Linear(hidden_size_etm, hidden_size_etm), \
                                             nn.BatchNorm1d(hidden_size_etm), nn.ReLU())
        self.category_out = nn.Linear(hidden_size_etm, category_size_o)
    def forward(self, category, attribute_output):
        batch_size = attribute_output.size(0)
        category_condition = torch.cat((attribute_output,torch.reshape(category, (batch_size,category_size_o))),1)
        
        category_output = self.category_linear_0(category_condition)
        category_output = self.category_norm(category_output)
        category_output = self.category_linear_1(category_output)
        category_output = self.category_linear_1(category_output)
        category_output = self.category_linear_1(category_output)
        category_output = self.category_out(category_output)
        category_output = torch.sigmoid(category_output)
        return category_output
        
class LengthGenerator(nn.Module):
    def __init__(self, length_size=1, attribute_size=12, hidden_size_length=256):
        super().__init__()
        self.length_linear_0 = nn.Sequential(nn.Linear(length_size+attribute_size, hidden_size_length), \
                               nn.BatchNorm1d(hidden_size_length), nn.ReLU())
        
        self.length_norm = nn.BatchNorm1d(hidden_size_length)
        self.length_linear_1 = nn.Sequential(nn.Linear(hidden_size_length, hidden_size_length), \
                               nn.BatchNorm1d(hidden_size_length), nn.ReLU())
        self.length_out = nn.Linear(hidden_size_length, length_size)
    def forward(self, lengths, attribute_output):
        batch_size = attribute_output.size(0)
        length_condition = torch.cat((attribute_output,torch.reshape(lengths, (batch_size,-1))),1)
        length_output = self.length_linear_0(length_condition)
        length_output = self.length_linear_1(length_output)
        length_output = self.length_linear_1(length_output)
        length_output = self.length_linear_1(length_output)
        length_output = self.length_linear_1(length_output)
        length_output = self.length_out(length_output)
        length_output = torch.sigmoid(length_output)
        return length_output


class FeatureGenerator(nn.Module):
    def __init__(self, in_size, out_size, lstm_layer=1, head_num=1, hidden_size=256):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hidden_size, lstm_layer, batch_first = True)
        self.query_linear = nn.Linear(hidden_size,hidden_size)                                                                                                                     
        self.key_linear = nn.Linear(hidden_size,hidden_size)                                                                                                                       
        self.value_linear = nn.Linear(hidden_size,hidden_size)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, head_num)
        self.out_layer = nn.Linear(hidden_size, out_size) 
    def forward(self, feature, kge_output, category_output, length_output, step_size=10.): 
        length_output = length_output*LENGTH_MAX
        lengths = torch.ceil(length_output/step_size)
        lengths[lengths==0] = 1
        condition_size = kge_output.size(1)+category_output.size(1)+length_output.size(1)
        
        feature_condition = torch.cat(\
                              (torch.cat((kge_output, category_output, length_output), 1)\
                               .repeat(1, feature.size(1))\
                               .view(feature.size(0),feature.size(1),condition_size),\
                               feature), 2)        
        packed_feature = rnn.pack_padded_sequence(feature_condition, lengths.transpose(1,0)[0], batch_first=True, enforce_sorted=False)  
        lstm_out,_ = self.lstm(packed_feature)
        hidden_seq, hidden_lengths = rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=feature.size(1)) #torch.Size([10, 310, 256])
        batch_size = hidden_seq.size(0)
        seq_len = hidden_seq.size(1)
        hidden_size = hidden_seq.size(2)
        query_proj = self.query_linear(hidden_seq.view(batch_size*seq_len,hidden_size))\
                         .view(batch_size,seq_len,hidden_size)\
                         .permute(1,0,2)
        key_proj = self.key_linear(hidden_seq.view(batch_size*seq_len,hidden_size))\
                       .view(batch_size,seq_len,hidden_size)\
                       .permute(1,0,2)
        value_proj = self.value_linear(hidden_seq.view(batch_size*seq_len,hidden_size))\
                         .view(batch_size,seq_len,hidden_size)\
                         .permute(1,0,2)       
        attn_output, attn_output_weights = self.multihead_attn(query_proj, key_proj, value_proj) 
        output = self.out_layer(attn_output.permute(1,0,2).contiguous().view(batch_size*seq_len,hidden_size))\
                     .view(batch_size,seq_len*out_size)  
        output = torch.sigmoid(output) 
        if output.max()>1:
            print('###output.max() is ###',output.max())                   

        output = output.view(batch_size,-1,3)
        
        pad_tag = torch.zeros(output.size(0), output.size(1))
        length_temp = torch.ceil(length_output.transpose(1,0)[0])
        for ii in np.arange(length_temp.size(0)):
            pad_tag[ii][0:int(length_temp[ii])] = 1
        if use_cuda:
            pad_tag = pad_tag.cuda(gpu)
        pad_tag = pad_tag.view(output.size(0),output.size(1),1).expand(-1,-1,output.size(2))        
        output = output*pad_tag
        if output.max()>1:
            print('###output.max() is ###',output.max())
        output = output.view(batch_size,seq_len*out_size)
        return output 


class Generator(nn.Module):
    def __init__(self, in_size, out_size, attribute_size, category_size_o, lstm_layer=1, head_num=1, hidden_size=256, \
                hidden_size_attri=256, hidden_size_etm=256, hidden_size_length=256):
        super().__init__()
        self.feature_generator = FeatureGenerator(in_size, out_size, lstm_layer, head_num, hidden_size)
        self.category_generator = CategoryGenerator(category_size_o, attribute_size+category_size_o, hidden_size_etm)
        self.length_generator = LengthGenerator(1,attribute_size, hidden_size_length)
    def forward(self, noise_attribute, noise_category, noise_length, noise_feature, step_size=10.): 
        attribute_output = noise_attribute
        category_output = self.category_generator(noise_category, attribute_output)
        length_output = self.length_generator(noise_length, attribute_output)
        feature_output = self.feature_generator(noise_feature, attribute_output, category_output, length_output, step_size)
        out_put = torch.cat((attribute_output, category_output, length_output, feature_output), 1)
        return out_put

class Discriminator(nn.Module):
    def __init__(self, in_size, hidden_size=256):
        super().__init__()
        self.linear_0 = nn.Sequential(nn.Linear(in_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU())
        self.linear_1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU())
        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, outputs):
        outputs = self.linear_0(outputs)                                                                                                                                                                                                                                                                                                                       
        outputs = self.linear_1(outputs)
        outputs = self.linear_1(outputs)
        outputs = self.linear_1(outputs)
        outputs = self.linear_1(outputs)        
        outputs = self.linear_out(outputs)
        return outputs
        
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def generate_data(netG, kge, gene_size=128):
    noise_attribute = kge
    noise_category = torch.rand(gene_size,category_size_o)
    noise_length = torch.rand(gene_size,1)
    noise_feature = torch.randn(gene_size, int(LENGTH_MAX/10), 30)
    if use_cuda:
        noise_attribute = noise_attribute.cuda(gpu)
        noise_category = noise_category.cuda(gpu)
        noise_length = noise_length.cuda(gpu)
        noise_feature = noise_feature.cuda(gpu)
    output = netG(noise_attribute, noise_category, noise_length, noise_feature)
    
    attribute_output, category_output, length_output, feature_output = torch.split(output, [attribute_size, category_size_o, 1, int(LENGTH_MAX*3)], dim=1)
    feature_output = feature_output.view(-1, int(LENGTH_MAX), 3)

    category_output = category_output.cpu().data.numpy()
    length_output = np.ceil(length_output.cpu().data.numpy()*LENGTH_MAX)
    feature_output = feature_output.cpu().data.numpy()
    
    return category_output, length_output, feature_output



BATCH_SIZE = 64
LENGTH_MAX = 500.
LAMBDA = 10
ITERS = 2000 
CRITIC_ITERS = 5

gene_size = 2048
                          
attribute_size = 128  #kge dim
category_size_o = 12
feature_size = int(LENGTH_MAX*3)
in_size = attribute_size+category_size_o+1+feature_size
out_size = 30


file_name = '/data/IoT/data.npz'
dataset=TrafficDataset(file_name)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

netG = Generator(attribute_size+category_size_o+1+out_size, out_size, attribute_size, category_size_o)
netD = Discriminator(in_size)
pretrained_netG = torch.load('/data/IoT/netG',map_location=torch.device('cpu')) # the pre-trained model
netG.load_state_dict(pretrained_netG)

save_dir = '/data/IoT/generated_data/' #save trained model and generated data here

#netG = torch.nn.DataParallel(netG, device_ids=[3,2,1])
#netD = torch.nn.DataParallel(netD, device_ids=[3,2,1])
#netG.to(f'cuda:{netG.device_ids[0]}')
#netD.to(f'cuda:{netD.device_ids[0]}')

if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)
print(netG)
print(netD)
    
optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.tensor(1, dtype=torch.float32)
mone = one * -1.0

if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)
for iteration in trange(ITERS):
    iter_d = 0
    for idx, data in enumerate(data_loader):

        for p in netD.parameters():  
            p.requires_grad = True  
        iter_d = iter_d + 1
        
        feature_batch = data[0].view(BATCH_SIZE, -1)
        category_batch = data[1]
        lengths_batch = data[2].view(BATCH_SIZE, -1)/LENGTH_MAX
        kge_batch = data[3]
        
        
        netD.zero_grad()
        
        real_data = torch.cat((kge_batch, category_batch, lengths_batch, feature_batch), 1)
        if use_cuda:
            real_data = real_data.cuda(gpu)
        D_real = netD(real_data)
        D_real = D_real.mean()
        D_real.backward(mone)
        
        
        noise_attribute = kge_batch
        noise_category = torch.rand(BATCH_SIZE,category_size_o)
        noise_length = torch.rand(BATCH_SIZE,1)
        noise_feature = torch.randn(BATCH_SIZE, int(LENGTH_MAX/10), 30)
        if use_cuda:
            noise_attribute = noise_attribute.cuda(gpu)
            noise_category = noise_category.cuda(gpu)
            noise_length = noise_length.cuda(gpu)
            noise_feature = noise_feature.cuda(gpu) 

        fake_data = netG(noise_attribute, noise_category, noise_length, noise_feature)
        D_fake = netD(fake_data)
        D_fake = D_fake.mean()
        D_fake.backward(one)
        
        gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data)
        gradient_penalty.backward()
        
        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()
            
        if iter_d%CRITIC_ITERS == 0:
            for p in netD.parameters():
                p.requires_grad = False  
            netG.zero_grad()
            
            noise_attribute = kge_batch
            noise_category = torch.rand(BATCH_SIZE,category_size_o)
            noise_length = torch.rand(BATCH_SIZE,1)
            noise_feature = torch.randn(BATCH_SIZE, int(LENGTH_MAX/10), 30)
            if use_cuda:
                noise_attribute = noise_attribute.cuda(gpu)
                noise_category = noise_category.cuda(gpu)
                noise_length = noise_length.cuda(gpu)
                noise_feature = noise_feature.cuda(gpu) 
            fake = netG(noise_attribute, noise_category, noise_length, noise_feature)
            G = netD(fake)
            G = G.mean()
            G.backward(mone)
            G_cost = -G
            optimizerG.step()

    if iteration % 50 == 0:
        kge_index = torch.randint(0, len(dataset), (gene_size, ))
        kge = dataset.kge[kge_index]
        category_output, length_output, feature_output = generate_data(netG, kge, gene_size)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez(os.path.join(save_dir+"/iteration-"+str(iteration), 'data_generated.npz'), \
                 data_feature=feature_output, \
                 data_length=length_output,\
                 data_category=category_output)
        torch.save(netD.state_dict(), os.path.join(save_dir+"/iteration-"+str(iteration), 'netD'))
        torch.save(netG.state_dict(), os.path.join(save_dir+"/iteration-"+str(iteration), 'netG'))

        
        
        
        
        
        
