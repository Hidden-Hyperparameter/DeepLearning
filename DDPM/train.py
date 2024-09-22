import sys
import os

# parent_dir = os.path.abspath('/root/DeepLearning')
# parent_dir = os.path.abspath('/home/zhh24/DeepLearning')
parent_dir = os.path.abspath('..')

sys.path.append(parent_dir)
print('appended',parent_dir)

import utils

from DDPM import DDPM
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.utils

import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)
mnist = utils.MNIST(batch_size=256)
train_loader = mnist.train_dataloader
valid_loader = mnist.valid_dataloader
T=1000
# beta1=1e-4 # variance of lowest temperature
# betaT=6e-2 # variance of highest temperature
eps = 8e-3
steps=torch.linspace(0,T,steps=T+1,dtype=torch.float)
f_t=torch.cos(((steps/T+eps)/(1.0+eps))*3.14159*0.5)**2
betas=torch.clamp(1.0-f_t[1:]/f_t[:T],0.0,0.999)

# step = torch.log(torch.tensor(betaT/beta1))/(T-1)
# betas = beta1 * torch.exp(step*torch.arange(T,dtype=torch.float).to(device))
# step = (betaT-beta1)/(T-1)
# betas = torch.arange(T,dtype=torch.float,device=device) * step + beta1


alphas = 1-betas
alpha_bars = alphas.clone()
for i in range(1,T):
    alpha_bars[i] *= alpha_bars[i-1]

print(alpha_bars)
print('range of bars',alpha_bars.min(),alpha_bars.max())
# print(alphas)

sqrt = torch.sqrt
sigmas = sqrt(betas * (1-alpha_bars / alphas)/(1-alpha_bars))
sigmas[0] = 1
print('range of sigmas,',sigmas.min(),sigmas.max())
alphas = alphas.to(device)
alpha_bars = alpha_bars.to(device)
betas = betas.to(device)
sigmas = sigmas.to(device)
weights = torch.ones(T,dtype=torch.float,device=device)

@torch.no_grad()
def sample(model:DDPM,save_dir):
    x = torch.randn([100,784]).to(device)
    for t in range(T-1,-1,-1):
        sigmaz = torch.randn_like(x)*sigmas[t]
        if t==0:
            sigmaz = 0
        noise_pred = model(x,t*torch.ones(x.shape[0],dtype=torch.long,device=device))
        x0_pred = (x - noise_pred * sqrt(1 - alpha_bars[t])) / sqrt(alpha_bars[t]).clamp(-1,1)
        mean_pred = (sqrt(alphas[t]) * (1-alpha_bars[t]/alphas[t]) * x + sqrt(alpha_bars[t]/alphas[t]) * (1-alphas[t]) * x0_pred) / (1-alpha_bars[t])
        x = mean_pred + sigmaz
        # x = torch.clamp(x,0,1)
    grid = torchvision.utils.make_grid(post_process(x).reshape(-1,1,28,28).cpu(), nrow=10)
    torchvision.utils.save_image(grid, save_dir)

@torch.no_grad()
def visualize(model,save_dir):
    interval = (T-1) // 20
    x = torch.randn([10,784]).to(device)
    x_history = []
    for t in range(T-1,-1,-1):
        sigmaz = torch.randn_like(x)*((betas[t])**0.5).to(device)
        if t==0:
            sigmaz = 0
        x = (x-(1-alphas[t])/(sqrt(1-alpha_bars[t]))*model(x,t*torch.ones(x.shape[0],dtype=torch.long,device=device)))/(sqrt(alphas[t]))+sigmaz
        # x = torch.clamp(x,0,1)
        x_history.append(x)
    # print('cat.shape',torch.cat(x_history,dim=0).shape)
    grid = torchvision.utils.make_grid(post_process(torch.stack(x_history,dim=0)[::interval,...]).reshape(-1,1,28,28).cpu(), nrow=10)
    torchvision.utils.save_image(grid, save_dir)
    print('Saved visualize to',os.path.abspath(save_dir))

@torch.no_grad()
def visualize_denoise(model,save_dir):
    # get 10 images from the dataset
    x,_ = next(iter(valid_loader))
    x = x[:20,...].reshape(20,784).to(device)
    x = pre_process(x)
    t = torch.tensor([i * T // 20 for i in range(20)],dtype=torch.long,device=device)
    noise = torch.randn_like(x).reshape(-1,784)
    v1 = (sqrt(alpha_bars[t]).reshape(-1,1)*x).reshape(-1,784)
    v2 = sqrt(1-alpha_bars[t]).reshape(-1,1)*noise
    x_corr = v1+v2
    est = model(x_corr,t)
    x_rec = (x_corr - sqrt(1-alpha_bars[t]).reshape(-1,1)*est)/(sqrt(alpha_bars[t])).reshape(-1,1)
    grid_orig = torchvision.utils.make_grid(post_process(x).reshape(-1,1,28,28).cpu(), nrow=10)
    grid_corr = torchvision.utils.make_grid(post_process(x_corr).reshape(-1,1,28,28).cpu(), nrow=10)
    grid_rec = torchvision.utils.make_grid(post_process(x_rec).reshape(-1,1,28,28).cpu(), nrow=10)
    # add noise level infomation to the image
    noise_level = (1-alpha_bars[t]).reshape(-1).tolist()
    ori_mse = noise.pow(2).mean(dim=1).reshape(-1).tolist()
    mse = ((est-noise)**2).mean(dim=1).reshape(-1).tolist()
    print(noise_level)
    print(ori_mse)
    print(mse)
    grid = torch.cat([grid_orig,grid_corr,grid_rec],dim=1)
    torchvision.utils.save_image(grid, save_dir)
    print('Saved denoise to',os.path.abspath(save_dir))

def plot_loss(losses,save_dir):
    losses_vals, t_vals = zip(*losses)
    losses_vals = torch.cat(losses_vals,dim=0)
    t_vals = torch.cat(t_vals,dim=0)

    results = []
    for t in range(T):
        this_t = abs(t_vals.float()-float(t))<0.5
        results.append(torch.sum(torch.where(this_t,losses_vals,torch.tensor(0.,device=device))).item() / (torch.sum(this_t.float())+1e-3).item())
    plt.plot(results)
    plt.ylim(0,max(results)* 1.2)
    plt.savefig(save_dir)
    plt.close()
    # weights = (torch.tensor(results,device=device)) # weights
    weights = torch.ones(T,dtype=torch.float,device=device)
    # weights[:10]=0
    # weights[10:80] /= 100
    return weights

def pre_process(x):
    return x*2-1

def post_process(x):
    return (x+1)/2

def train(epochs,model:DDPM,optimizer,eval_interval=5):
    global weights
    for epoch in range(epochs):
        # print('weights normalized:',weights/weights.sum())
        all_ts = torch.distributions.Categorical(weights).sample((50000,))
        cnt = 0
        model.train()
        with tqdm(train_loader) as bar:
            losses = []
            for x,_ in bar:
                cnt += x.shape[0]
                x = pre_process(x.to(device))
                epss = torch.randn_like(x).reshape(-1,784).to(device)
                # ts = torch.randint(0,T,(x.shape[0],),device=device,dtype=torch.long)
                ts = all_ts[cnt-x.shape[0]:cnt]
                alpha_tbars = alpha_bars[ts]
                value = (sqrt(alpha_tbars).reshape(-1,1,1,1)*x).reshape(-1,784)+sqrt(1-alpha_tbars).reshape(-1,1)*epss
                out = model(value,ts) # [batch,784]
                # loss = ((epss-out).pow(2).mean(dim=-1) * (betas[ts])/(2*alphas[ts]*(1-alpha_tbars))).sum(dim=0)
                loss = ((epss-out).pow(2).mean(dim=-1)).mean(dim=0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                bar.set_description('epoch {}, loss {:.4f}'.format(epoch,sum(losses)/len(losses)))

        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader) as bar:
                mses = []
                losses = []
                losses_for_t = []
                for x,_ in bar:
                    x = pre_process(x.to(device))
                    epss = torch.randn_like(x).reshape(-1,784).to(device)
                    ts = torch.randint(0,T,(x.shape[0],),device=device,dtype=torch.long)
                    # print(ts)
                    alpha_tbars = alpha_bars[ts]
                    value = (sqrt(alpha_tbars).reshape(-1,1,1,1)*x).reshape(-1,784)+sqrt(1-alpha_tbars).reshape(-1,1)*epss
                    out = model(value,ts)
                    mse = F.mse_loss(epss,out)
                    mses.append(mse.item())
                    loss = ((epss-out).pow(2).mean(dim=-1))
                    # loss = (epss-out).pow(2).mean(dim=-1)
                    losses_for_t.append((loss.clone().detach(),ts))
                    loss = (loss).mean(dim=0)
                    losses.append(loss.item())
                    bar.set_description('epoch {}, MSE {:.4f}, [Valid] {:.4f}'.format(epoch,sum(mses)/len(mses),sum(losses)/len(losses)))
                    
        if epoch % eval_interval == 0:
            visualize(model,save_dir=os.path.join('./samples',f'diffuse_epoch_{epoch}.png'))
            sample(model,save_dir=os.path.join('./samples',f'sample_epoch_{epoch}.png'))
            # visualize_denoise(model,save_dir=os.path.join('./samples',f'denoise_epoch_{epoch}.png'))
            weights = plot_loss(losses_for_t,save_dir=os.path.join('./samples',f'loss_epoch_{epoch}.png'))
            torch.save(model,os.path.join('./samples',f'epoch_{epoch}.pt'))

if __name__ == '__main__':
    model = DDPM().to(device)
    print('Number parameters of the model:', sum(p.numel() for p in model.parameters()))
    print('Model strcuture:',model)
    optimizer = torch.optim.Adam(model.parameters(),lr=2e-4)
    os.makedirs('./samples',exist_ok=True)
    # sample(model,save_dir=os.path.join('./samples',f'init.png'))
    # visualize(model,save_dir=os.path.join('./samples',f'init_visualize.png'))
    train(200,model,optimizer,eval_interval=5)