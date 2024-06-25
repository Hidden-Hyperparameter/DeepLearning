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

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

mnist = utils.MNIST(batch_size=128)
train_loader = mnist.train_dataloader
valid_loader = mnist.valid_dataloader

@torch.no_grad()
def sample(model:VAE,save_dir):
    pass

def train(epochs,model:VAE,optimizer,eval_interval=1):
    for epoch in range(epochs):
        losses = []
        with tqdm(train_loader) as bar:
            for i,(x,y) in enumerate(train_loader):
                loss = model.get_loss(x,y)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 1000 == 0:
                    bar.set_description('epoch {}: loss {:.4f}'.format(epoch,sum(losses[:-100])/100))

if __name__ == '__main__':
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    sample(model,save_dir=os.path.join('./VAE/samples',f'init.png'))
    train(100,model,optimizer,eval_interval=1)