#!/usr/bin/env python3


"""Train or retrain a model."""


import sys
import random
import io
import os
import argparse
import subprocess

from tqdm import tqdm, trange

import numpy as np
import pandas as pd
import PIL
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter

from structure import SelfNLLLoss, SelfPixelwiseNLLLoss, FullNet
from utils import img_to_tensor, jpeg_compress 

writer = SummaryWriter('runs/Adaptive CFA Forensics')

def preprocess(img_name, quality=None):
    """Reads an image, applies JPEG compression if specified, and convert it to a torch tensor."""
    img = plt.imread(img_name)
    Y, X, C = img.shape
    img = img[:Y-Y%2, :X-X%2, :3]
    if quality is not None:
        img = jpeg_compress(img, quality)
    img = img_to_tensor(img).cuda().type(torch.float)
    return img


def train_net(images, net, lr=1e-3, n_epochs_auxiliary=1000, n_epochs_blockwise=500, batch_size=20, block_size=32, save_path='trained_model.pt'):
    """Trains a network on images, and save the network.
    :param images: list of torch.tensor on CUDA, each of shape (1, 4, Y, X) (Y and X can be different accross images)
    :param net: nn.Module, network to train or retrain.
    :param n_epochs_auxiliary: int, number of epochs on the auxiliary training net. Default: 1000
    :param n_epochs_blockwise: int, number of epochs on the final blockwise net. Default: 500
    :param batch_size: int. Default: 20
    :param block_size: int. Default: 32
    :param save_path: string, where to write the trained model. Default: 'trained_model.pt'.
    """
    #writer = SummaryWriter('runs/Adaptive CFA Forensics')
    running_loss = 0.0
    criterion = SelfPixelwiseNLLLoss().cuda()
    optim = torch.optim.Adam(net.auxiliary.parameters(), lr=lr)
    optim.zero_grad()
    for epoch in trange(n_epochs_auxiliary):
        random.shuffle(images)
        for i_img, img in enumerate(images):
            o = net.auxiliary(img)
            loss = criterion(o, global_best=True)
            loss.backward()
            running_loss += loss.item()
            
            if (i_img+1) % batch_size == 0:
                optim.step()
                optim.zero_grad()
                writer.add_scalar('training loss auxiliary', running_loss/1000, epoch)
                running_loss = 0.0
            
    first_processor = nn.Sequential(net.spatial, net.pixelwise, net.grids, nn.AvgPool2d(block_size))
    images = [torch.tensor(first_processor(img).detach().cpu().numpy()).cuda() for img in images]  #Make sure no gradient stays
    criterion = SelfNLLLoss().cuda()
    optim = torch.optim.Adam(net.blockwise.parameters(), lr=lr)
    optim.zero_grad()
    running_loss = 0.0
    for epoch in trange(n_epochs_blockwise):
        random.shuffle(images)
        for i_img, img in enumerate(images):
            o = net.blockwise(img)
            loss = criterion(o, global_best=True)
            loss.backward()
            running_loss += loss.item()
            if (i_img+1) % batch_size == 0:
                optim.step()
                optim.zero_grad()
                writer.add_scalar('training loss blockwise', running_loss/1000, epoch)
                running_loss = 0.0
    torch.save(net.state_dict(), save_path)
    

def get_parser():
    parser = argparse.ArgumentParser(description="Train model on multiple images.")
    parser.add_argument("-m", "--model", type=str, default=None, help="If a model is specified, initialise the network with given model. If nothing is specified, the network is randomly initialised.")
    parser.add_argument("-j", "--jpeg", type=int, default=None, help="JPEG compression quality. Default: no compression is done before analysis.")
    parser.add_argument("-b", "--block-size", type=int, default=32, help="Block size. Default: 32.")
    parser.add_argument("-o", "--out", type=str, default="trained_model.pt", help="Where to save the trained model. Default: trained_model.pt")
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-3, help="Learning rate. Default: 1e-3.")
    parser.add_argument("-a", "--epochs-auxiliary", type=int, default=1000, help="Number of epochs for the auxiliary network. Default: 1000.")
    parser.add_argument("-B", "--epochs-blockwise", type=int, default=500, help="Number of epochs for the blockwise network. Default: 500.")
    parser.add_argument("-s", "--batch-size", type=int, default=15, help="Batch size. Default: 15.")
    parser.add_argument("input", nargs='+', type=str, help="Images to analyse.")
    return parser

def create_image_names_arr(path,tail_img):
    """Duyệt file hình ảnh, kiểm tra xem có phải hình ảnh hay không rồi add vào mảng"""
    FJoin = os.path.join
    img_names = [FJoin(path, f) for f in os.listdir(path)]
    arr_names = []
    for name in img_names:
       if(tail_img in name):
          arr_names.append(name)
    return arr_names


def get_dataset_info(list_txt):
    status = []
    demosaic_1 = []
    demosaic_2 = []
    au_images = []
    for filename in list_txt:
      res = subprocess.run(['cat', filename], stdout=subprocess.PIPE).stdout.decode('utf-8')
      res = res.split('\n')
      if len(res) == 2: #Authentic
        status.append(res[0])
        demosaic_1.append(res[1])
        demosaic_2.append(None)
        au_images.append(None)
      else:
        status.append(res[0])
        demosaic_1.append(res[1].split(' ')[0])
        demosaic_2.append(res[1].split(' ')[1])
        au_images.append(res[2])
      fixed_name = []
    for item in list_txt:
      fixed_name.append(item.split('/')[-1].replace('_gt.txt', '.png'))
    dataset_info = pd.DataFrame({'name': fixed_name,'status': status, 'demosaic_1': demosaic_1, 'demosaic_2': demosaic_2, 'au_images': au_images})
    return dataset_info

def get_random_img(path, dataset_info):
    demosaic_select =['ha','bilinear','lmmse']
    data_select = dataset_info[dataset_info['demosaic_1'].isin(demosaic_select)]
    img_names = random.sample(list(data_select['name']), 19)
    FJoin = os.path.join
    img_names_dir = [FJoin(path, f) for f in img_names]
    return img_names_dir


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    out = args.out
    path_input = args.input[0]
    txt_names_fill = create_image_names_arr(path_input, 'txt')
    data_info = get_dataset_info(txt_names_fill)
    img_names = get_random_img(path_input, data_info)
    print(img_names)
    print(len(img_names))
    quality = args.jpeg
    images = [preprocess(img_name, quality=quality) for img_name in img_names]
    lr = args.learning_rate
    n_epochs_auxiliary = args.epochs_auxiliary
    n_epochs_blockwise = args.epochs_blockwise
    batch_size = args.batch_size
    block_size = args.block_size
    model_path = args.model
    net = FullNet().cuda()
    if model_path is not None:
        net.load_state_dict(torch.load(model_path))
    train_net(images, net, lr, n_epochs_auxiliary, n_epochs_blockwise, batch_size, block_size, out)

    
