from model import Flow
device = 'cuda'
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
    if not os.path.exists('./impainting'):
        os.makedirs('./impainting')
    torchvision.utils.save_image(grid,os.path.join('./impainting',f'corrupted_level{level}.png'))
    grid = torchvision.utils.make_grid(x.reshape(-1,1,28,28).cpu(), nrow=N)
    torchvision.utils.save_image(grid,os.path.join('./impainting',f'ground_truth.png'))
    print('corrption MSE:',mse_loss(x,x_c).item())
    return x,x_c,rnd_mask

def impainting(model:Flow,steps=10000,lr=0.1,max_norm=50):
    model.eval()
    truth,x,mask = gen_corrpted_sample()
    x_for_calc = model.pre_process(x)
    x_for_calc.requires_grad_(True)
    opt = torch.optim.Adam([x_for_calc],lr=lr)
    with torch.no_grad():
        print('original log prob:',-model.get_loss(truth).item())
        print('corrption log prob:',-model.get_loss(x).item())
    # x_for_calc = x.clone().detach()
    # opt = torch.optim.Adam([x_for_calc],lr=lr)
    # opt.zero_grad()
    print(x_for_calc.shape)
    up, down = x_for_calc[:,:,:14,:],x_for_calc[:,:,14:,:]
    up.requires_grad_(True)
    down.requires_grad_(True)
    optim = torch.optim.Adam([down],lr=lr)
    with tqdm(range(steps)) as bar:
        for step in bar:
            # temp = x_for_calc.clone()
            optim.zero_grad()
            # x_for_calc.requires_grad_(True)
            total = torch.cat([up,down],dim=2)
            loss = model.get_loss(total,pre_process=False)
            loss.backward()
            # print(x_for_calc[:,:,14:,:].grad.norm())
            optim.step()
            # print(torch.nn.functional.mse_loss(x_for_calc,old).item())
            total = torch.cat([up,down],dim=2)
            x = torch.sigmoid(total).clone().detach()
            mse = mse_loss(x,truth)
            bar.set_description(f'[Impainting], step {step}, mse loss {mse.item()}, log prob {-loss.item()}')
            # if step > 10:
                # exit()
            # time.sleep(0.5)
            if (step+1) % 10 == 0:
                grid = torchvision.utils.make_grid(x.reshape(-1,1,28,28).cpu(), nrow=5)
                torchvision.utils.save_image(grid,os.path.join('./impainting',f'rec_step_{step}.png'))

if __name__ == '__main__':
    model = torch.load('./samples/epoch_38.pt').to(device)
    model.requires_grad_(False)
    # model.eval()
    impainting(model)