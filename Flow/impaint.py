from model import Flow,device
from train import valid_loader
import torch,torchvision,os
from tqdm import tqdm
from torch.nn.functional import mse_loss

def gen_corrpted_sample(level=0.3) -> torch.Tensor:
    x = next(iter(valid_loader))[0].to(device)
    x = x[:25,...]

    rnd_mask = torch.zeros_like(x)
    rnd_mask[...,:14,:]=1

    # mask = 1: cannot change.
    x_c = x*(rnd_mask) + (level * torch.randn_like(x).to(x.device))*(1-rnd_mask)
    x_c = x_c.clamp(0,1)

    # x_c[...,14:,:]=0
    # rnd_mask[...,14:,:]=0

    grid = torchvision.utils.make_grid(x_c.reshape(-1,1,28,28).cpu(), nrow=5)
    torchvision.utils.save_image(grid,os.path.join('./impainting',f'corrupted_level{level}.png'))
    grid = torchvision.utils.make_grid(x.reshape(-1,1,28,28).cpu(), nrow=5)
    torchvision.utils.save_image(grid,os.path.join('./impainting',f'ground_truth.png'))
    print('corrption MSE:',mse_loss(x,x_c).item())
    return x,x_c,rnd_mask

def impainting(model:Flow,steps=1000,lr=0.1):
    truth,x,mask = gen_corrpted_sample()
    print('corrption log prob:',-model.get_loss(x).item())
    with tqdm(range(steps)) as bar:
        for step in bar:
            x.requires_grad_(True)
            if x.grad is not None:
                x.grad.zero_()
            loss = model.get_loss(x) # negative log likelihood
            loss.backward()
            with torch.no_grad():
                # print((lr*x.grad).norm())
                # print(x.norm())
                if x.grad.norm() > 10:
                    x.grad /= ( x.grad.norm()/10)
                x = x - lr * x.grad * (1-mask)

                x = x.clamp(0,1)
                mse = mse_loss(x,truth)
            bar.set_description(f'[Impainting], step {step}, mse loss {mse.item()}, log prob {-loss.item()}')
            if (step+1) % 100 == 0:
                grid = torchvision.utils.make_grid(x.reshape(-1,1,28,28).cpu(), nrow=5)
                torchvision.utils.save_image(grid,os.path.join('./impainting',f'rec_step_{step}.png'))

if __name__ == '__main__':
    model = torch.load('./samples/epoch_20.pt').to(device)
    model.requires_grad_(False)
    model.eval()
    impainting(model)