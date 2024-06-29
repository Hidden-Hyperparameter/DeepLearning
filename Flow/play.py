from model import Flow,device
import torch,torchvision
epoch = 4
dataset = torch.load(f'./samples/dataset_1.pt').to(device)
model = torch.load(f'./samples/epoch_{epoch}.pt')
model.eval()
z = dataset[:100,...]
# z = z + torch.randn_like(z)*0.5
real_x = model.backward(z)
print(z.mean(dim=0).max(),z.mean(dim=0).min())
print(z.std(dim=0).max(),z.std(dim=0).min())
gaussian = torch.distributions.Normal(
    loc=z.mean(dim=0),
    scale=z.std(dim=0)
)
zp = gaussian.sample([100])
# zp = z.mean(dim=0).unsqueeze(0).repeat(100,1)


# zp = z+torch.randn_like(z)*0.5
# zp = (z+torch.cat((z[::2],z[1::2]),dim=0))/2
real_x_p = model.backward(zp)

grid = torchvision.utils.make_grid(real_x.reshape(-1,1,28,28).cpu(), nrow=10)
torchvision.utils.save_image(grid, f'./play/epoch_{epoch}.png')

grid = torchvision.utils.make_grid(real_x_p.reshape(-1,1,28,28).cpu(), nrow=10)
torchvision.utils.save_image(grid, f'./play/resample_epoch_{epoch}.png')