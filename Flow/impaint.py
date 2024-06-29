from model import Flow,device
from train import valid_loader
import torch,torchvision,os
from tqdm import tqdm
import sys,time
sys.path.append(os.path.abspath('..'))
from torch.nn.functional import mse_loss
torch.manual_seed(42)

def gen_corrpted_sample(level=0.3) -> torch.Tensor:
    N = 10
    x = next(iter(valid_loader))[0].to(device)
    x = x[:N*N,...]

    rnd_mask = torch.zeros_like(x)
    rnd_mask[...,:14,:]=1

    # mask = 1: cannot change.
    x_c = x*(rnd_mask) + (level * torch.randn_like(x).to(x.device))*(1-rnd_mask)
    x_c = x_c.clamp(0,1)

    # x_c[...,14:,:]=0
    # rnd_mask[...,14:,:]=0

    grid = torchvision.utils.make_grid(x_c.reshape(-1,1,28,28).cpu(), nrow=N)
    torchvision.utils.save_image(grid,os.path.join('./impainting',f'corrupted_level{level}.png'))
    grid = torchvision.utils.make_grid(x.reshape(-1,1,28,28).cpu(), nrow=N)
    torchvision.utils.save_image(grid,os.path.join('./impainting',f'ground_truth.png'))
    print('corrption MSE:',mse_loss(x,x_c).item())
    return x,x_c,rnd_mask

def impainting(model:Flow,steps=1000,lr=0.001):
    model.eval()
    # model.train()
    truth,x,mask = gen_corrpted_sample()
    # x_for_calc = torch.sigmoid(x)
    x_for_calc = x.clone().detach()
    with torch.no_grad():
        print('corrption log prob:',-model.get_loss(x_for_calc).item())
    # x_for_calc = x.clone().detach()
    with tqdm(range(steps)) as bar:
        for step in bar:
            x_for_calc.requires_grad_(True)
            if x_for_calc.grad is not None:
                x_for_calc.grad.zero_()
            loss = model.get_loss(x_for_calc) # negative log likelihood
            loss.backward()
            with torch.no_grad():
                norm_clip = x_for_calc.grad.reshape(x_for_calc.shape[0],-1).norm(dim=-1).clamp(5,1e5)/5
                x_for_calc.grad /= norm_clip.view(-1,1,1,1)
                change = lr * x_for_calc.grad * (1-mask)
                x_for_calc = x_for_calc - change
                x_for_calc = x_for_calc.clamp(1e-5,1-1e-5)
                # x = torch.logit(x_for_calc)
                x = x_for_calc.clone().detach()
                mse = mse_loss(x,truth)
            bar.set_description(f'[Impainting], step {step}, mse loss {mse.item()}, log prob {-loss.item()}')
            # if step > 10:
                # exit()
            # time.sleep(0.5)
            if (step+1) % 100 == 0:
                grid = torchvision.utils.make_grid(x.reshape(-1,1,28,28).cpu(), nrow=10)
                torchvision.utils.save_image(grid,os.path.join('./impainting',f'rec_step_{step}.png'))

if __name__ == '__main__':
    model = torch.load('./samples/epoch_8.pt').to(device)
    model.requires_grad_(False)
    # model.eval()
    impainting(model)