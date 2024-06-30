from model import Flow,device
from train import valid_loader
import torch,torchvision,os
from tqdm import tqdm
import sys,time
sys.path.append(os.path.abspath('..'))
from torch.nn.functional import mse_loss
torch.manual_seed(42)

def gen_corrpted_sample(level=0.3) -> torch.Tensor:
    N = 5
    x = next(iter(valid_loader))[0].to(device)
    x = x[:N*N,...]

    rnd_mask = torch.zeros_like(x)
    rnd_mask[...,:14,:]=1
    # rnd_mask = (torch.rand_like(x) < 0.2).float()

    # mask = 1: cannot change.
    x_c = x*(rnd_mask) + (level * torch.randn_like(x).to(x.device))*(1-rnd_mask)
    x_c = x_c.clamp(0,1)

    grid = torchvision.utils.make_grid(x_c.reshape(-1,1,28,28).cpu(), nrow=N)
    torchvision.utils.save_image(grid,os.path.join('./impainting',f'corrupted_level{level}.png'))
    grid = torchvision.utils.make_grid(x.reshape(-1,1,28,28).cpu(), nrow=N)
    torchvision.utils.save_image(grid,os.path.join('./impainting',f'ground_truth.png'))
    print('corrption MSE:',mse_loss(x,x_c).item())
    return x,x_c,rnd_mask

def impainting(model:Flow,steps=10000,lr=0.1,max_norm=5):
    model.eval()
    # model.train()
    truth,x,mask = gen_corrpted_sample()
    x_for_calc = (x+1e-5).log()-(1-x+1e-5).log()
    with torch.no_grad():
        print('original log prob:',-model.get_loss(truth).item())
        print('corrption log prob:',-model.get_loss(x).item())
    # x_for_calc = x.clone().detach()
    # opt = torch.optim.Adam([x_for_calc],lr=lr)
    # opt.zero_grad()
    with tqdm(range(steps)) as bar:
        for step in bar:
            # opt.zero_grad()
            x_for_calc.requires_grad_(True)
            x_for_calc.grad = torch.zeros_like(x_for_calc)
            old = x_for_calc.clone()
            loss = model.get_loss(x_for_calc,pre_process=False) # negative log likelihood
            loss.backward()
            grad = x_for_calc.grad.reshape(x.shape[0],-1)
            grad_norm = grad.norm(dim=-1,keepdim=True).clamp(max_norm,1e5)
            grad /= (grad_norm/max_norm)    
            x_for_calc.grad = grad.reshape_as(x_for_calc)     
            # opt.step()
            x_for_calc = (x_for_calc - lr * x_for_calc.grad) * (1-mask) + x_for_calc * mask
            x_for_calc = x_for_calc.clamp(-20,20).clone().detach()
            # print(torch.nn.functional.mse_loss(x_for_calc,old).item())
            x = torch.sigmoid(x_for_calc).clone().detach()
            mse = mse_loss(x,truth)
            bar.set_description(f'[Impainting], step {step}, mse loss {mse.item()}, log prob {-loss.item()}')
            # if step > 10:
                # exit()
            # time.sleep(0.5)
            if (step+1) % 100 == 0:
                grid = torchvision.utils.make_grid(x.reshape(-1,1,28,28).cpu(), nrow=5)
                torchvision.utils.save_image(grid,os.path.join('./impainting',f'rec_step_{step}.png'))

if __name__ == '__main__':
    model = torch.load('./samples/epoch_77.pt').to(device)
    model.requires_grad_(False)
    # model.eval()
    impainting(model)