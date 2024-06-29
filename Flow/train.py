import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(parent_dir)

import utils_2

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np

from model import Flow,device,float_tp


mnist = utils_2.MNIST(batch_size=500)
train_loader = mnist.train_dataloader
valid_loader = mnist.valid_dataloader


@torch.no_grad()
def sample(model,save_dir):
    model.eval()
    hiddens = torch.randn([100,784]).to(device).to(float_tp)
    outs = model.backward(hiddens)
    # use torchvision to display 100 figures in 10 * 10 grid
    grid = torchvision.utils.make_grid(outs.reshape(-1,1,28,28).cpu(), nrow=10)
    torchvision.utils.save_image(grid, save_dir)

# def train(epochs,model,optimizer,eval_interval=1):
#     for epoch in range(epochs):
#         losses = []

#         model.train()
#         dataset = []
#         with tqdm(valid_loader) as bar:
#             for x,y in bar:
#                 x = x.to(device).to(float_tp)
#                 with torch.no_grad():
#                     zi,_ = model(x)
#                     dataset.append(zi.clone())
#                 loss = model.get_loss(x)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 losses.append(loss.item())

#                 if len(losses) > 10:
#                     bar.set_description(f'Epoch: {epoch} Loss: {sum(losses[-10:])/10}')
#         with torch.no_grad():
#             dataset = torch.cat(dataset,dim=0)
#             torch.save(dataset,os.path.join('./samples',f'dataset_{epoch}.pt'))

#         losses = []
#         model.eval()
#         with torch.no_grad():
#             with tqdm(valid_loader) as bar:
#                 for x,y in bar:
#                     x = x.to(device).to(float_tp)
#                     loss = model.get_loss(x)
#                     losses.append(loss.item())
#                     bar.set_description(f'Epoch: {epoch} [Valid]Loss: {sum(losses)/len(losses)}')
        

#         if epoch % eval_interval == 0:
#             model.eval()
#             with torch.no_grad():
#                 # print('statics on real data:')
#                 try:
#                     z
#                 except NameError:
#                     pass
#                 else:
#                     rec = model.backward(z)
#                     grid = torchvision.utils.make_grid(rec.reshape(-1,1,28,28).cpu(), nrow=16)
#                     torchvision.utils.save_image(grid, os.path.join('./samples',f'rec_oldz_epoch_{epoch}.png'))
                

#                 z,logdet = model(x)
#                 rec = model.backward(z)
#                 grid = torchvision.utils.make_grid(rec.reshape(-1,1,28,28).cpu(), nrow=16)
#                 torchvision.utils.save_image(grid, os.path.join('./samples',f'rec_valid_epoch_{epoch}.png'))
                
#                 # x = x[:1,...]
#                 # z,logdet = model(x)
#                 # batch = x.shape[0]
#                 # x = x.reshape(batch,1,-1).expand(batch,784,784) 
#                 # logdet = logdet[0]
#                 # print('logdet:',logdet.item())
#                 # # add is the identity matrix
#                 # eps = 0.1
#                 # add = torch.eye(784).to(device)*eps
#                 # xp = x + torch.cat((add.unsqueeze(0),torch.zeros(batch-1,784,784).to(x.device)),dim=0) # shape: [[batch,784,784]
#                 # zp = model(xp.reshape(-1,1,28,28))[0].reshape(batch,784,784) # first 784 means which dim to disturb
#                 # mat = (zp[0]-z[0].reshape(1,784))
#                 # param = (abs(mat)+1e-5).log().mean()
#                 # print('param:',param)
#                 # mat/=(torch.exp(param))
#                 # print(mat,param)
#                 # det = torch.linalg.det(mat)
#                 # print(det)
#                 # print('real log determinant',param*784+torch.log(det).item()-np.log(eps)*784)

#                 # print(logdet.mean())
#                 # print(z[0].norm())
#                 # print('loss:',z[0].norm()**2/2-logdet[0])

#                 # print('statics on perturbed data:')
#                 # # rec = model.backward(z)
#                 # x_change = (torch.randn_like(x)*0.1 + x).clamp(0,1)
#                 # z,logdet = model(x_change)
#                 # print(logdet.mean())
#                 # print(z[0].norm())
#                 # print('loss:',z[0].norm()**2/2-logdet[0])


                
#                 # levels = [0.001,0.01,0.1]
#                 # for level in levels:
#                 #     preturbed_z = z + level * torch.randn_like(z)
#                 #     rec = model.backward(preturbed_z)
#                 #     grid = torchvision.utils.make_grid(rec.reshape(-1,1,28,28).cpu(), nrow=16)
#                 #     torchvision.utils.save_image(grid, os.path.join('./samples',f'preturblevel{level}_epoch_{epoch}.png'))


            # sample(model,save_dir=os.path.join('./samples',f'epoch_{epoch}.png'))
            # torch.save(model,os.path.join('./samples',f'epoch_{epoch}.pt'))

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
        rec = model.backward(z)
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

    utils_2.count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=3e-5)
    sample(model,save_dir=os.path.join('./samples',f'init.png'))
    train(100,model,optimizer,eval_interval=1)