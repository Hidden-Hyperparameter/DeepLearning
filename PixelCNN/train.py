import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(parent_dir)

import utils

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np

from pixelcnn import PixelCNN,device


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

def train(epochs,model:PixelCNN,optimizer,eval_interval=1):
    for epoch in range(epochs):
        losses = []

        model.train()
        with tqdm(train_loader) as bar:
            for x,y in bar:
                x = x.to(device)
                loss = model.get_loss(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                bar.set_description(f'Epoch: {epoch} Loss: {sum(losses)/len(losses)}')
       
        losses = []
        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader) as bar:
                for x,y in bar:
                    x = x.to(device)
                    loss = model.get_loss(x)
                    losses.append(loss.item())
                    bar.set_description(f'Epoch: {epoch} [Valid]Loss: {sum(losses)/len(losses)}')

        if (epoch+1) % eval_interval == 0:
            sample(model,save_dir=os.path.join('./samples',f'epoch_{epoch}.png'))
            # torch.save(model,os.path.join('./samples',f'epoch_{epoch}.pt'))

if __name__ == '__main__':
    model = PixelCNN()
    model.to(device)
    utils.count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=5e-5)
    # sample(model,save_dir=os.path.join('./samples',f'init.png'))
    train(100,model,optimizer,eval_interval=1)