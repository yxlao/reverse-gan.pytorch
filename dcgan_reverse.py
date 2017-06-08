from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import time
from torch.autograd import Variable
import numpy as np


from dcgan import NetG

def reverse_gan(opt):
    ngpu = 1
    nz = 100
    ngf = 64
    nc = 3
    batch_size = 1
    cuda = True
    z_distribution = 'uniform'

    torch.cuda.set_device(0)

    # load netG
    netG = NetG(ngpu, nz, ngf, nc)
    netG.load_state_dict(torch.load(opt.netG))

    # generate ground-truth noise
    if z_distribution == 'uniform':
        noise = torch.FloatTensor(batch_size, nz, 1, 1).uniform_(-1, 1)
    elif z_distribution == 'normal':
        noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
    else:
        raise ValueError()
    noise = Variable(noise)
    noise.data.resize_(batch_size, nz, 1, 1)
    noise.data.normal_(0, 1)

    for param in netG.parameters():
        param.requires_grad = False

    # fix fake, and try to find noise_approx
    if z_distribution == 'uniform':
        noise_approx = torch.FloatTensor(batch_size, nz, 1, 1).uniform_(-1, 1)
    elif z_distribution == 'normal':
        noise_approx = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
    else:
        raise ValueError()

    mse_loss = nn.MSELoss()
    mse_loss_ = nn.MSELoss()

    if cuda:
        netG.cuda()
        mse_loss.cuda()
        mse_loss_.cuda()
        noise, noise_approx = noise.cuda(), noise_approx.cuda()

    noise_approx = Variable(noise_approx)
    noise_approx.requires_grad = True

    # generate ground-truth fake
    fake = netG(noise)

    optimizer_approx = optim.Adam([noise_approx], lr=0.01, betas=(opt.beta1, 0.999))

    for i in range(100000):
        fake_approx = netG(noise_approx)
        mse_g_z = mse_loss(fake_approx, fake)
        mse_z = mse_loss_(noise_approx, noise)
        if i % 100 == 0:
            print("[Iter {}] MSE_FAKE: {}, MSE_Z: {}".format(i, mse_g_z.data[0],
                                                             mse_z.data[0]))

        optimizer_approx.zero_grad()
        mse_g_z.backward()
        optimizer_approx.step()

    vutils.save_image(fake.data, 'fake.png', normalize=True)
    vutils.save_image(fake_approx.data, 'fake_approx.png', normalize=True)

    print(noise.cpu().data.numpy().squeeze())
    print(noise_approx.cpu().data.numpy().squeeze())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_distribution', default='uniform',
                        help='uniform | normal')
    parser.add_argument('--nz', type=int, default=100,
                        help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=100000,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--netG', default='dcgan_out/netG_epoch_10.pth',
                        help="path to netG (to continue training)")
    parser.add_argument('--outf', default='dcgan_out',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--profile', action='store_true',
                        help='enable cProfile')

    opt = parser.parse_args()
    print(opt)
    reverse_gan(opt)
