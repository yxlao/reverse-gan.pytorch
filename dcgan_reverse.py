from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from dcgan import NetG


def reverse_z(netG, g_z, z, opt, clip='disabled'):
    """
    Estimate z_approx given G and G(z).

    Args:
        netG: nn.Module, generator network.
        g_z: Variable, G(z).
        opt: argparse.Namespace, network and training options.
        z: Variable, the ground truth z, ref only here, not used in recovery.

    Returns:
        Variable, z_approx, the estimated z value.
    """
    # sanity check
    assert clip in ['disabled', 'standard', 'stochastic']

    # loss metrics
    mse_loss = nn.MSELoss()
    mse_loss_ = nn.MSELoss()

    # init tensor
    if opt.z_distribution == 'uniform':
        z_approx = torch.FloatTensor(1, opt.nz, 1, 1).uniform_(-1, 1)
    elif opt.z_distribution == 'normal':
        z_approx = torch.FloatTensor(1, opt.nz, 1, 1).normal_(0, 1)
    else:
        raise ValueError()

    # transfer to gpu
    if opt.cuda:
        mse_loss.cuda()
        mse_loss_.cuda()
        z_approx = z_approx.cuda()

    # convert to variable
    z_approx = Variable(z_approx)
    z_approx.requires_grad = True

    # optimizer
    optimizer_approx = optim.Adam([z_approx], lr=opt.lr,
                                  betas=(opt.beta1, 0.999))

    # train
    for i in range(opt.niter):
        g_z_approx = netG(z_approx)
        mse_g_z = mse_loss(g_z_approx, g_z)
        mse_z = mse_loss_(z_approx, z)
        if i % 100 == 0:
            print("[Iter {}] mse_g_z: {}, MSE_z: {}"
                  .format(i, mse_g_z.data[0], mse_z.data[0]))

        # bprop
        optimizer_approx.zero_grad()
        mse_g_z.backward()
        optimizer_approx.step()

        # clipping
        if clip == 'standard':
            z_approx.data[z_approx.data > 1] = 1
            z_approx.data[z_approx.data < -1] = -1
        if clip == 'stochastic':
            z_approx.data[z_approx.data > 1] = random.uniform(-1, 1)
            z_approx.data[z_approx.data < -1] = random.uniform(-1, 1)

    # save g(z_approx) image
    vutils.save_image(g_z_approx.data, 'g_z_approx.png', normalize=True)

    return z_approx


def reverse_gan(opt):
    # load netG and fix its weights
    netG = NetG(opt.ngpu, opt.nz, opt.ngf, opt.nc)
    netG.load_state_dict(torch.load(opt.netG))
    for param in netG.parameters():
        param.requires_grad = False

    # init z
    if opt.z_distribution == 'uniform':
        z = torch.FloatTensor(1, opt.nz, 1, 1).uniform_(-1, 1)
    elif opt.z_distribution == 'normal':
        z = torch.FloatTensor(1, opt.nz, 1, 1).normal_(0, 1)
    else:
        raise ValueError()
    z = Variable(z)
    z.data.resize_(1, opt.nz, 1, 1)

    # transfer to gpu
    if opt.cuda:
        netG.cuda()
        z = z.cuda()

    # generate g_z
    g_z = netG(z)
    vutils.save_image(g_z.data, 'g_z.png', normalize=True)
    print(z.cpu().data.numpy().squeeze())

    # recover z_approx from standard
    z_approx = reverse_z(netG, g_z, z, opt, clip='stochastic')
    print(z_approx.cpu().data.numpy().squeeze())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_distribution', default='uniform',
                        help='uniform | normal')
    parser.add_argument('--nz', type=int, default=100,
                        help='size of the latent z vector')
    parser.add_argument('--nc', type=int, default=3,
                        help='number of channels in the generated image')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=10000,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use')
    parser.add_argument('--netG', default='dcgan_out/netG_epoch_10.pth',
                        help="path to netG (to continue training)")
    parser.add_argument('--outf', default='dcgan_out',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--profile', action='store_true',
                        help='enable cProfile')

    opt = parser.parse_args()
    print(opt)

    # process arguments
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run "
              "with --cuda")

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True  # turn on the cudnn autotuner
        # torch.cuda.set_device(1)

    reverse_gan(opt)
