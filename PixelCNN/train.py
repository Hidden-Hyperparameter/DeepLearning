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
def sample(model:PixelCNN | GatedPixelCNN,save_dir):
    model.eval()
    imgs = model.sample(batch=100)
    utils.save_figure(
        path=save_dir,
        image=imgs,
        nrow=10,
    )

def visualize(model):
    """used to debug, avoid model cheating"""
    model.eval()
    x = torch.zeros(1,1,28,28).to(device)
    x.requires_grad = True
    y = model(x,torch.zeros(x.shape[0]).to(device))
    y[...,14,14].sum().backward()
    # show heatmap
    heatmap = ((x.grad[0,0,...].abs()>1e-4).float()*0.5).cpu().numpy()
    plt.imshow(heatmap,cmap='hot')
    # plot a star at (2,2)
    plt.scatter(14,14,c='blue',marker='*',s=100)
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
    # choice = 'PixelCNN'
    choice = 'GatedPixelCNN'
    model = dic[choice]['model']()
    model.to(device)
    utils.count_parameters(model)
    info = {
        'lr':1e-3,
        'weight_decay':1e-4,
    }
    optimizer = torch.optim.Adam(model.parameters(),**info)
    print('optimizer info:',info)
    sample(model,save_dir=os.path.join(f'./samples_{choice}',f'init.png'))
    # visualize(model); exit()
    train(100,model,optimizer,eval_interval=1,sample_func=sample,train_loader=train_loader,valid_loader=valid_loader,save_dir=f'./samples_{choice}',conditional=dic[choice]['conditional'])