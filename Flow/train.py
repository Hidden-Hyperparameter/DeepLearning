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

from model import Flow,device,float_tp


mnist = utils.MNIST(batch_size=500)
train_loader = mnist.train_dataloader
valid_loader = mnist.valid_dataloader


@torch.no_grad()
def sample(model,save_dir):
    model.eval()
    hiddens = torch.randn([100,784]).to(device).to(float_tp)
    outs = model.zhhbackward(hiddens)
    # use torchvision to display 100 figures in 10 * 10 grid
    grid = torchvision.utils.make_grid(outs.reshape(-1,1,28,28).cpu(), nrow=10)
    torchvision.utils.save_image(grid, save_dir)

def train(epochs,model,optimizer,eval_interval=1):
    for epoch in range(epochs):
        losses = []

        model.train()
        with tqdm(train_loader) as bar:
            for x,y in bar:
                x = x.to(device)
                
                loss = model.get_loss(x)

                optimizer.zero_grad()
                # print('loss:',loss)
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

        z,logdet = model(x)
        rec = model.zhhbackward(z)
        grid = torchvision.utils.make_grid(rec.reshape(-1,1,28,28).cpu(), nrow=32)
        torchvision.utils.save_image(grid, os.path.join('./samples',f'rec_valid_epoch_{epoch}.png'))
        
        if (epoch+1) % eval_interval == 0:
            sample(model,save_dir=os.path.join('./samples',f'epoch_{epoch}.png'))
            torch.save(model,os.path.join('./samples',f'epoch_{epoch}.pt'))

if __name__ == '__main__':
    model = Flow()
    # print(list(model.modules()))
    for module in model.modules():
        if isinstance(module,(nn.Linear,nn.Conv2d)):
            # nn.init.orthogonal_(module.weight)
            nn.init.xavier_normal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

    model.to(device)

    utils.count_parameters(model)
    # optimizer = torch.optim.Adam(model.parameters(),lr=6e-5,weight_decay=5e-5)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=5e-5)
    if not os.path.exists('./samples'):
        os.makedirs('./samples')
    sample(model,save_dir=os.path.join('./samples',f'init.png'))
    train(100,model,optimizer,eval_interval=1)