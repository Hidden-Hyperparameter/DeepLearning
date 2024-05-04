import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(parent_dir)

import utils

from DDPM import DDPM
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.utils

mnist = utils.MNIST()
train_loader = mnist.train_dataloader
T=1000
beta1=1e-4
betaT=2e-2
step = (betaT-beta1)/(T-1)
betas = torch.arange(beta1,betaT+step,step)
alphas = 1-betas
alpha_bars = alphas[:]
for i in range(1,T):
    alpha_bars[i] *= alpha_bars[i-1]


@torch.no_grad()
def sample(model:DDPM,save_dir):
    x = torch.randn([100,784])
    for t in range(T,0,-1):
        z = torch.randn_like(x)*((betas[t])**0.5)
        if t==1:
            z = 0
        x = 1/((alphas[t])**0.5)*(x-(1-alphas[t])/((1-alpha_bars[t])**0.5)*model(x,t))+z
    grid = torchvision.utils.make_grid(x, nrow=10)
    torchvision.utils.save_image(grid, save_dir)

def train(epochs,model:DDPM,optimizer,eval_interval=1):
    for epoch in range(epochs):
        with tqdm(train_loader) as bar:
            for x,_ in bar:
                epss = torch.randn_like(x).reshape(-1,784)
                ts = (torch.rand(x.shape[0])*T).to(torch.long)//T+1
                alpha_tbars = alpha_bars[ts]
                # print(x.shape)
                out = model((((alpha_tbars)**0.5).reshape(-1,1,1,1)*x).reshape(-1,784)+((1-alpha_tbars)**0.5).reshape(-1,1)*epss,ts)
                loss = F.mse_loss(epss,out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.set_description('epoch {}, loss {:.4f}'.format(epoch,loss))
        if epoch % eval_interval == 0:
            sample(model,save_dir=os.path.join('samples',f'epoch_{epoch}.png'))

if __name__ == '__main__':
    model = DDPM()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    train(10,model,optimizer)