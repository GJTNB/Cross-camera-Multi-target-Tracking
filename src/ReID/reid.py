# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2

from .model import ft_net
#####################################################################
#Show result
def imshow(path, title=None, cnt=-1):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################

def load_network(name, network):
    save_path = os.path.join('./ReID/model',name,'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders, ms):

    det_feat = [None] * 2
    count = 0
    for j, data in enumerate(dataloaders):
        features = None
        features = torch.FloatTensor()
        img = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n,512).zero_().cuda()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
        det_feat[j] = features 
    return det_feat


###load config###
# load the training config
def ReID(cam_id, i, model, data_transforms, ms):
    g = "gallery_" + cam_id
    data_dir = "./data/"
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in [g,'query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                shuffle=False, num_workers=0) for x in [g,'query']}

    gallery_path = image_datasets[g].imgs


    ######################################################################
    # Load Collected data Trained model

    # Extract feature
    with torch.no_grad():
        gallery_feature = extract_feature(model,dataloaders[g], ms)
        query_feature = extract_feature(model,dataloaders['query'], ms)

    query = query_feature.view(-1,1)
    # print(query.shape)
    score = torch.mm(gallery_feature,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    max_score = score.max()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]

    query_path, _ = image_datasets['query'].imgs[i]
    img_path, _ = image_datasets[g].imgs[index[0]]
    # print(img_path)
    num = index[0]
    # print(img_path)
    return num, max_score
