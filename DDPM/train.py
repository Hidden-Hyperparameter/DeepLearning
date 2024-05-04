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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist = utils.MNIST(batch_size=128)
train_loader = mnist.train_dataloader
valid_loader = mnist.valid_dataloader
T=100
beta1=1e-4
betaT=2e-2
step = (betaT-beta1)/(T-1)
betas = torch.arange(beta1,betaT+step,step).to(device)
alphas = 1-betas
alpha_bars = alphas[:]
for i in range(1,T):
    alpha_bars[i] *= alpha_bars[i-1]


@torch.no_grad()
def sample(model:DDPM,save_dir):
    x = torch.randn([100,784]).to(device)
    for t in range(T-1,-1,-1):
        sigmaz = torch.randn_like(x)*((betas[t])**0.5).to(device)
        if t==1:
            sigmaz = 0
        x = 1/((alphas[t])**0.5)*(x-(1-alphas[t])/((1-alpha_bars[t])**0.5)*model(x,(t*torch.ones(x.shape[0],dtype=torch.long)).to(device)))+sigmaz
    # x = torch.sigmoid(x)
    grid = torchvision.utils.make_grid(x.reshape(-1,1,28,28).cpu(), nrow=10)
    torchvision.utils.save_image(grid, save_dir)

def train(epochs,model:DDPM,optimizer,eval_interval=1):
    for epoch in range(epochs):
        model.train()
        with tqdm(train_loader) as bar:
            losses = 0
            nums = 0
            for x,_ in bar:
                x = x.to(device)
                # print(x)
                epss = torch.randn_like(x).reshape(-1,784).to(device)
                ts = ((torch.rand([x.shape[0]])*T).to(torch.long)).to(device)
                alpha_tbars = alpha_bars[ts]
                # print(x.shape)
                value = (((alpha_tbars)**0.5).reshape(-1,1,1,1)*x).reshape(-1,784)+((1-alpha_tbars)**0.5).reshape(-1,1)*epss
                # print(x)
                # print(value.shape)
                # print(epss.shape)
                # print((value - epss).mean())
                out = model(value,ts)
                # if epoch >= 2:
                #     # print('epss',epss[0])
                #     print('out',out[0])
                    # print('minus',epss[0]-out[0])
                    # print('num',(out[0]>1e-4).sum())
                loss = ((epss-out)**2).mean().sum()
                # F.mse_loss(epss,out)
                losses += loss
                nums += x.shape[0]
                optimizer.zero_grad()
                loss.backward()
                # print(model.res_blocks[0].reses[0].conv1[0].weight)
                # print('grad',model.res_blocks[0].reses[0].conv1[0].weight.grad)
                optimizer.step()
                bar.set_description('epoch {}, loss {:.4f}'.format(epoch,128*losses/nums))
        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader) as bar:
                losses = 0
                nums = 0
                for x,_ in bar:
                    x = x.to(device)
                    epss = torch.randn_like(x).reshape(-1,784).to(device)
                    # ts = ((torch.rand([1])*T*torch.ones(x.shape[0])).to(torch.long)//T+1).to(device)
                    ts = ((torch.rand(x.shape[0])*T).to(torch.long)).to(device)
                    alpha_tbars = alpha_bars[ts]
                    value = (((alpha_tbars)**0.5).reshape(-1,1,1,1)*x).reshape(-1,784)+((1-alpha_tbars)**0.5).reshape(-1,1)*epss
                    out = model(value,ts)
                    loss = F.mse_loss(epss,out)
                    losses += loss.item()
                    nums += x.shape[0]
                    bar.set_description('epoch {}, valid loss {:.4f}'.format(epoch,128*losses/nums))
        if epoch % eval_interval == 0:
            sample(model,save_dir=os.path.join('./DDPM/samples',f'new_epoch_{epoch}.png'))
            torch.save(model,os.path.join('./DDPM/samples',f'new_epoch_{epoch}.pt'))

if __name__ == '__main__':
    model = DDPM().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    sample(model,save_dir=os.path.join('./DDPM/samples',f'init.png'))
    train(100,model,optimizer,eval_interval=1)