# from dataset import train_loader, test_loader
import sys,os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import utils
mnist = utils.MNIST(batch_size=128)
train_loader = mnist.train_dataloader
test_loader = mnist.valid_dataloader
from model import VQVAE

import torch,tqdm,os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Your device is:',device)
model = VQVAE().to(device)
x = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {x:,} trainable parameters')
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

@torch.no_grad()
def entropy(index):
    # index.shape: [batch,9,9]
    batch = index.shape[0]
    entropies = torch.zeros(index.shape[1:]).to(index.device)
    for i in range(128):
        counts = (index == i).sum(dim=0)
        entropies += (counts / batch) * torch.log(counts / batch + 1e-7)
    return -entropies.mean()

def avg(array):
    return sum(array)/len(array)

def train(epochs=20):

    for epoch in range(epochs):

        # train
        model.train()
        loss_recs = []
        dict_losses = []
        enc_losses = []
        var_losses = []
        bar = tqdm.tqdm(train_loader)
        data = []
        for x,_ in bar:
            x = x.to(device)

            loss_rec,dict_loss,enc_loss,var_loss,index = model.update(x)
            loss_recs.append(loss_rec.item())
            dict_losses.append(dict_loss.item())
            enc_losses.append(enc_loss.item())
            var_losses.append(var_loss.item())
            data.append(index.cpu())

            mean_1 = avg(loss_recs)
            mean_2 = avg(dict_losses)
            mean_3 = avg(enc_losses)
            mean_4 = avg(var_losses)
            bar.set_description(f'Epoch {epoch}/{epochs}, Loss: {(mean_1+mean_2+mean_3+mean_4):.4f}, Rec: {mean_1:.4f}, Dict: {mean_2:.4f}, Enc: {mean_3:.4f}, Var: {mean_4:.4f}')
        torch.save(torch.cat(data,dim=0),os.path.join('models',f'data_ep{epoch:02d}.pt'))
        bar.close()

        torch.save(model.state_dict(),os.path.join('models',f'vqvae_ep{1 + epoch:02d}.pth'))

        # eval
        model.eval()
        with torch.no_grad():
            loss_recs = []
            dict_losses = []
            enc_losses = []
            indexes = []
            var_losses = []
            for x,_ in test_loader:
                x = x.to(device)
                loss_rec,dict_loss,enc_loss,var_loss,index = model.update(x,do_update=False)
                loss_recs.append(loss_rec.item())
                dict_losses.append(dict_loss.item())
                enc_losses.append(enc_loss.item())
                var_losses.append(var_loss.item())
                indexes.append(index)
            mean_1 = avg(loss_recs)
            mean_2 = avg(dict_losses)
            mean_3 = avg(enc_losses)
            mean_4 = avg(var_losses)
            print(f'[Valid] Epoch {epoch}/{epochs}, Loss: {(mean_1+mean_2+mean_3+mean_4):.4f}, Rec: {mean_1:.4f}, Dict: {mean_2:.4f}, Enc: {mean_3:.4f}, Var: {mean_4:.4f}, Ent: {entropy(torch.cat(indexes,dim=0)).item():.4f}')
        
        # view reconstruction images
        latents = index
        images = model.generate(latents).cpu()
        import torchvision.utils as vutils
        vutils.save_image(images, os.path.join('samples',f'rec_{1 + epoch:02d}.png'))
    

if __name__ == '__main__':
    train()