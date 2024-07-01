import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(parent_dir)

import utils
from utils import train_generative_model as train

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np

from pixelcnn import PixelCNN,device
from gatedpixelcnn import GatedPixelCNN

mnist = utils.MNIST(batch_size=256)
train_loader = mnist.train_dataloader
valid_loader = mnist.valid_dataloader


@torch.no_grad()
def sample(model:PixelCNN,save_dir):
    model.eval()
    imgs = model.sample(100)
    utils.save_figure(
        path=save_dir,
        image=imgs,
        nrow=10,
    )

def show_reception_field(model):
    model.eval()
    x = torch.zeros(1,1,28,28).to(device)
    x.requires_grad = True
    y = model(x,torch.zeros(x.shape[0]).to(device))
    y[...,14,14].sum().backward()
    # show heatmap
    heatmap = ((x.grad[0,0,...].abs()>1e-4).float()*0.5).cpu().numpy()
    heatmap[14,14]=1
    plt.imshow(heatmap,cmap='hot')
    # save heatmap
    plt.savefig('./heatmap.png')

dic = {
    'PixelCNN':{
        'model':PixelCNN,
        'conditional':False,
    },
    'GatedPixelCNN':{
        'model':GatedPixelCNN,
        'conditional':True,
    }
}

if __name__ == '__main__':
    choice = 'GatedPixelCNN'
    model = dic[choice]['model']()
    model.to(device)
    utils.count_parameters(model)
    info = {
        'lr':1e-3,
        'weight_decay':3e-5,
    }
    optimizer = torch.optim.Adam(model.parameters(),**info)
    print('optimizer info:',info)
    # sample(model,save_dir=os.path.join('./samples',f'init.png'))
    show_reception_field(model)
    train(100,model,optimizer,eval_interval=1,sample_func=sample,train_loader=train_loader,valid_loader=valid_loader,save_dir=f'./samples_{choice}',conditional=dic[choice]['conditional'])