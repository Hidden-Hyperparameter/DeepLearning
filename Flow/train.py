import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(parent_dir)

import utils

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.utils

from model import Flow,device


mnist = utils.MNIST(batch_size=128)
train_loader = mnist.train_dataloader
valid_loader = mnist.valid_dataloader


@torch.no_grad()
def sample(model,save_dir):
    hiddens = torch.randn([100,784]).to(device)
    outs = model.backward(hiddens)
    # use torchvision to display 100 figures in 10 * 10 grid
    grid = torchvision.utils.make_grid(outs.reshape(-1,1,28,28).cpu(), nrow=10)
    torchvision.utils.save_image(grid, save_dir)

def train(epochs,model,optimizer,eval_interval=1):
    for epoch in range(epochs):
        losses = []
        with tqdm(train_loader) as bar:
            for x,y in bar:
                x = x.to(device)
                loss = model.get_loss(x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                bar.set_description(f'Epoch: {epoch} Loss: {sum(losses[-10:])/10}')
        
        if epoch % eval_interval == 0:
            sample(model,save_dir=os.path.join('./samples',f'epoch_{epoch}.png'))
            torch.save(model,os.path.join('./samples',f'epoch_{epoch}.pt'))

if __name__ == '__main__':
    model = Flow().to(device)
    utils.count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    sample(model,save_dir=os.path.join('./samples',f'init.png'))
    train(100,model,optimizer,eval_interval=1)