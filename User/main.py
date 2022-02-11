import os, sys, time
import matplotlib
matplotlib.use('SVG')
import torch
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.autograd as autograd
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from tqdm import trange
import setproctitle
setproctitle.setproctitle("traffic_gene")
from models import MLPGenerator
from models import MarkovSwitchGenerator
from models import PatternGenerator
from models import TCNDiscriminator
from models import UserTrafficDataset

gpu = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
torch.manual_seed(7)
CUDA = torch.cuda.is_available()

BATCH_SIZE = 256*8
NOISE_SIZE = 32
NUM_CLUSTERS = 6
LENGTH = 288
PATTERN_NUM = 6

LR = 1e-5
LR_S = 1e-2
B1 = 0.5
B2 = 0.9
Br = 1e2
Bc = 1
Ba = 1e-19
Bs = 1
ITERS = 10001
CRITIC_ITERS = 1
MAX_NORM_GRAD = 3

DIFF_CTR = torch.tensor(10.0)

hidden_sizes_mlp = [6 * 12, 6 * 12, 6 * 24, 6 * 24, 6 * 48, 6 * 48]

save_dir_head = '/data/user_gene_results/clu_gen/'
sub_dir = '20220211'
tb_writer = SummaryWriter()

data_filename = '/data/utf_dataset_file.npz'
dataset = UserTrafficDataset(data_filename, dense=True)
print('The number of samples: ', len(dataset))
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)

gpu = 0
net_G_list = []
for ii in range(NUM_CLUSTERS):
    netG = MarkovSwitchGenerator(pattern_num=PATTERN_NUM)
    net_G_list.append(netG)
netPatternG = PatternGenerator(pattern_num=PATTERN_NUM,
                               input_size=48,
                               noise_size=NOISE_SIZE,
                               out_seq_size=LENGTH,
                               hidden_size=24,
                               condition_size=32,
                               dropout=0.3,
                               num_layers=3)
netD = TCNDiscriminator(input_size=1,
                        num_clusters=NUM_CLUSTERS,
                        num_channels=None,
                        condition_size=32,
                        condition_squeeze_size=4,
                        dropout=0.3)

adversarial_loss = torch.nn.MSELoss()
cluster_loss = torch.nn.MSELoss()#CrossEntropyLoss()#
aggregate_loss = torch.nn.MSELoss()
smooth_loss = torch.nn.MSELoss()

if CUDA:
    for ii in range(NUM_CLUSTERS):
        net_G_list[ii] = net_G_list[ii].cuda(gpu)
    netPatternG = netPatternG.cuda(gpu)
    netD = netD.cuda(gpu)
    adversarial_loss = adversarial_loss.cuda(gpu)
    cluster_loss = cluster_loss.cuda(gpu)
    smooth_loss = smooth_loss.cuda(gpu)
    DIFF_CTR = DIFF_CTR.cuda(gpu)
optimizer_G_list = []
for ii in range(NUM_CLUSTERS):
    optimizer_G = torch.optim.Adam(net_G_list[ii].parameters(), lr=LR_S, betas=(B1, B2))
    optimizer_G_list.append(optimizer_G)
optimizer_PatternG = torch.optim.Adam(netPatternG.parameters(), lr=LR, betas=(B1, B2))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=LR, betas=(B1, B2))

if True:
    for iteration in trange(ITERS):
        start_time = time.time()
        for idx, (user_id, app_usage, geo_sta, utf_norm, utf, _) in enumerate(data_loader):
            sample_num = len(user_id)
            real_criterion = torch.zeros(sample_num, requires_grad=False).cuda(gpu) if CUDA \
                else torch.zeros(sample_num, requires_grad=False)
            fake_criterion = torch.ones(sample_num, requires_grad=False).cuda(gpu) if CUDA \
                else torch.ones(sample_num, requires_grad=False)

            conditions = torch.cat((app_usage, geo_sta), 1).cuda(gpu) if CUDA \
                else torch.cat((app_usage, geo_sta), 1)
            real_data = utf_norm.cuda(gpu) if CUDA else utf_norm


            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True
            for p in netPatternG.parameters():
                p.requires_grad = False
            for ii in range(NUM_CLUSTERS):
                for p in net_G_list[ii].parameters():
                    p.requires_grad = False
            optimizer_D.zero_grad()
            D_real, D_weight = netD(real_data, conditions)
            D_real_clusters = torch.argmax(D_real[:, 1:], dim=1)

            noise = torch.randn(sample_num, NOISE_SIZE, requires_grad=False).cuda(gpu) if CUDA \
                else torch.randn(sample_num, NOISE_SIZE, requires_grad=False)
            fake_data_switch = torch.zeros([real_data.shape[0],PATTERN_NUM, real_data.shape[1]], requires_grad=False).cuda(gpu) if CUDA \
                else torch.zeros([real_data.shape[0],PATTERN_NUM, real_data.shape[1]], requires_grad=False)
            D_loss_clu = torch.zeros(1).cuda(gpu) if CUDA else torch.zeros(1)
            for ii in range(NUM_CLUSTERS):
                if (D_real_clusters==ii).any():
                    fake_data_switch[D_real_clusters==ii] = net_G_list[ii](noise[D_real_clusters==ii],
                                                                    conditions[D_real_clusters==ii])
                    real_data_center = real_data[D_real_clusters == ii].mean(0)
                    D_loss_clu = D_loss_clu \
                                 + cluster_loss(real_data[D_real_clusters == ii],
                                                real_data_center.expand_as(real_data[D_real_clusters == ii])) \
                                 - cluster_loss(real_data[D_real_clusters != ii],
                                                real_data_center.expand_as(real_data[D_real_clusters != ii]))
            fake_data_pattern = netPatternG(noise, conditions)
            fake_data = torch.mul(fake_data_switch, fake_data_pattern).sum(1)
            D_fake, D_weight = netD(fake_data.detach(), conditions)

            II = torch.eye(D_weight.shape[0], requires_grad=False).cuda(gpu) if CUDA \
                    else torch.eye(D_weight.shape[0], requires_grad=False)
            D_loss_regularity = torch.dist(D_weight, II)

            D_loss_adv = (adversarial_loss(D_real[:, 0], real_criterion)
                          + adversarial_loss(D_fake[:, 0], fake_criterion))/2
            D_loss = D_loss_adv + D_loss_clu * Bc + D_loss_regularity * Br
            D_loss.backward()
            torch.nn.utils.clip_grad_norm_(netD.parameters(), MAX_NORM_GRAD, norm_type=2)
            optimizer_D.step()
            tb_writer.add_scalar('loss\D\D_loss', D_loss, iteration)
            tb_writer.add_scalar('loss\D\D_loss_adv', D_loss_adv, iteration)
            tb_writer.add_scalar('loss\D\D_loss_clu', D_loss_clu, iteration)
            tb_writer.add_scalar('loss\D\D_loss_regularity', D_loss_regularity, iteration)

            if idx % CRITIC_ITERS == 0:
                for p in netD.parameters():  # reset requires_grad
                    p.requires_grad = False
                for p in netPatternG.parameters():
                    p.requires_grad = True
                for ii in range(NUM_CLUSTERS):
                    for p in net_G_list[ii].parameters():
                        p.requires_grad = True
                optimizer_PatternG.zero_grad()
                D_real, D_weight = netD(real_data, conditions)
                D_real_clusters = torch.argmax(D_real[:, 1:], dim=1)
                noise = torch.randn(sample_num, NOISE_SIZE, requires_grad=False).cuda(gpu) if CUDA \
                    else torch.randn(sample_num, NOISE_SIZE, requires_grad=False)
                fake_data_switch = torch.zeros([real_data.shape[0], PATTERN_NUM, real_data.shape[1]]).cuda(gpu) if CUDA \
                    else torch.zeros([real_data.shape[0], PATTERN_NUM, real_data.shape[1]])
                for ii in range(NUM_CLUSTERS):
                    if (D_real_clusters == ii).any():
                        optimizer_G_list[ii].zero_grad()
                        fake_data_switch[D_real_clusters == ii] = net_G_list[ii](noise[D_real_clusters == ii],
                                                                      conditions[D_real_clusters == ii])
                G_loss_sctn = torch.linalg.vector_norm(torch.diff(fake_data_switch))
                fake_data_pattern = netPatternG(noise, conditions)
                fake_data = torch.mul(fake_data_switch, fake_data_pattern).sum(1)
                D_fake, D_weight = netD(fake_data, conditions)
                D_fake_clusters = torch.argmax(D_fake[:, 1:], dim=1)

                G_loss_list = torch.zeros(NUM_CLUSTERS).cuda(gpu) if CUDA else torch.zeros(NUM_CLUSTERS)
                G_loss_agg_list = torch.zeros(NUM_CLUSTERS).cuda(gpu) if CUDA else torch.zeros(NUM_CLUSTERS)
                G_loss = torch.zeros(1).cuda(gpu) if CUDA else torch.zeros(1)
                for ii in range(NUM_CLUSTERS):
                    idx_clu = (D_real_clusters == ii)
                    if idx_clu.any():
                        gene_criterion = torch.ones_like(idx_clu, dtype=torch.float32, requires_grad=False)[idx_clu].cuda(gpu) \
                            if CUDA else torch.ones_like(idx_clu, dtype=torch.float32, requires_grad=False)[idx_clu]
                        G_loss_list[ii] = -torch.log(D_fake[idx_clu, ii+1] + 1e-9).sum() #CrossEntropyLoss
                        idx_clu_fake = (D_fake_clusters == ii)
                        if idx_clu_fake.any():
                            G_loss_agg_list[ii] = aggregate_loss(real_data[D_real_clusters == ii].sum(0),
                                                             fake_data[D_fake_clusters == ii].sum(0))
                        G_loss = G_loss + G_loss_list[ii] + G_loss_agg_list[ii] * Ba
                if G_loss_sctn >= 32:
                    G_loss = G_loss + G_loss_sctn * Bs
                G_loss.backward()
                for ii in range(NUM_CLUSTERS):
                    torch.nn.utils.clip_grad_norm_(net_G_list[ii].parameters(), MAX_NORM_GRAD, norm_type=2)
                    optimizer_G_list[ii].step()
                    tb_writer.add_scalar('loss\G_loss\G_loss_'+str(ii), G_loss_list[ii], iteration)
                    tb_writer.add_scalar('loss\G_loss\G_loss_aggregate_'+str(ii), G_loss_agg_list[ii], iteration)
                optimizer_PatternG.step()
                tb_writer.add_scalar('loss\G_loss\G_loss_sctn', G_loss_sctn, iteration)
                tb_writer.add_scalar('loss\G_loss\G_loss', G_loss, iteration)

                save_dir = save_dir_head + '/' + sub_dir + '/iteration-' + str(iteration)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                fake_data_np = fake_data.cpu().detach().numpy()
                real_data_np = real_data.cpu().detach().numpy()
                fig, ax = plt.subplots(figsize=(24, 16))
                n_bins = 100
                line_w = 2
                use_cumulative = -1
                use_log = True
                n_real, bins, patches = ax.hist(real_data_np.flatten(), n_bins, density=True,
                                                histtype='step', cumulative=use_cumulative, label='real', log=use_log,
                                                facecolor='g', linewidth=line_w)
                n_gene, bins, patches = ax.hist(fake_data_np.flatten(), n_bins, density=True,
                                                histtype='step', cumulative=use_cumulative, label='gene', log=use_log,
                                                facecolor='b', linewidth=line_w)
                ax.grid(True)
                ax.legend(loc='right')
                ax.set_title('Cumulative step histograms')
                ax.set_xlabel('Value')
                ax.set_ylabel('Likelihood of occurrence')
                plt.savefig(os.path.join(save_dir, 'fig_hist.jpg'))
                plt.close()
                dst = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)
                tb_writer.add_scalar('loss\G_loss\dst', dst, iteration)
        if iteration % 10 == 0:  # True
            fake_data_np = fake_data.cpu().detach().numpy()
            real_data_np = real_data.cpu().detach().numpy()

            torch.save(netD.state_dict(), os.path.join(save_dir, 'netD'), _use_new_zipfile_serialization=False)
            torch.save(netPatternG.state_dict(), os.path.join(save_dir, 'netPatternG'), _use_new_zipfile_serialization=False)
            np.savez(os.path.join(save_dir, 'gene_data'), gene_data = fake_data_np)
            for ii in range(NUM_CLUSTERS):
                torch.save(net_G_list[ii].state_dict(), os.path.join(save_dir, 'netG_'+str(ii)), _use_new_zipfile_serialization=False)

            fig_r_sum = plt.figure(figsize=(24, 16))
            plt.plot(real_data_np.sum(0))
            tb_writer.add_figure('fig\fig_r\fig_r_sum', fig_r_sum, iteration)
            plt.close()
            fig_r_samples = plt.figure(figsize=(24, 16))
            plt.plot(real_data_np[0::256])
            tb_writer.add_figure('fig\fig_r\fig_r_samples', fig_r_samples, iteration)
            plt.close()
            fig_f_sum =  plt.figure(figsize=(24, 16))
            plt.plot(fake_data_np.sum(0))
            tb_writer.add_figure('fig\fig_f\fig_f_sum', fig_f_sum, iteration)
            plt.close()
            fig_f_samples =  plt.figure(figsize=(24, 16))
            plt.plot(fake_data_np[0::256])
            tb_writer.add_figure('fig\fig_f\fig_f_samples', fig_f_samples, iteration)
            plt.close()

tb_writer.close()





