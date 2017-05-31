# GAN implementation
# - Use data reader from DCGGAN-tensorflow
# - Network / training scheme from pytorch dcgan example

from __future__ import print_function
import argparse
import os
import numpy as np
import tensorflow as tf
from dataset import MnistDataIter

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--batchSize', type=int, default=64,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')


if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)
