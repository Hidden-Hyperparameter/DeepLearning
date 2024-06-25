import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(parent_dir)

import utils

from VAE import VAE
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

mnist = utils.MNIST(batch_size=128)
train_loader = mnist.train_dataloader
valid_loader = mnist.valid_dataloader

@torch.no_grad()
def sample(model:VAE,save_dir):
    lables = (torch.arange(0,10).reshape([1,-1]) * torch.ones([10,1])).reshape(-1).to(device)
    lables = lables.to(torch.long)
    zs = torch.randn([100,model.latent_size]).to(device)
    outs = model.decoder(zs,lables)
    grid = torchvision.utils.make_grid(outs.reshape(-1,1,28,28).cpu(), nrow=10)
    torchvision.utils.save_image(grid, save_dir)

def train(epochs,model:VAE,optimizer,eval_interval=1):
    for epoch in range(epochs):
        reconstructions = []
        kls = []
        # lable_losses = []
        with tqdm(train_loader) as bar:
            for i,(x,y) in enumerate(bar):
                x,y = x.to(device),y.to(device)
                rec,kl = model.get_loss(x,y)
                # label_loss *= 100
                reconstructions.append(rec.item())
                kls.append(kl.item())
                # lable_losses.append(label_loss.item())
                
                optimizer.zero_grad()
                (rec + kl).backward()
                optimizer.step()

                if i % 10 == 0:
                    bar.set_description('epoch {}: rec {:.4f}, kl {:.4f}'.format(epoch,sum(reconstructions[-10:])/10,sum(kls[-10:])/10))#,sum(lable_losses[-10:])/10))
        if (epoch+1) % eval_interval == 0:
            sample(model,save_dir=os.path.join('./samples',f'epoch_{epoch+1}.png'))
            torch.save(model,os.path.join('./samples',f'epoch_{epoch+1}.pt'))
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = VAE().to(device)
    print(f'The model has {count_parameters(model)} parameters.')

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    sample(model,save_dir=os.path.join('./samples',f'init.png'))
    train(100,model,optimizer,eval_interval=5)