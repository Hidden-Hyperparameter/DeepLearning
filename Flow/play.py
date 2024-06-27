from model import Flow,device
import torch,torchvision
epoch = 16
dataset = torch.load(f'./samples/dataset_16.pt').to(device)
model = torch.load(f'./samples/epoch_{epoch}.pt')
model.eval()
z = dataset[:100,...]
z = z + torch.randn_like(z)*0.1
real_x = model.backward(z)
# print(z.mean(dim=0).max(),z.mean(dim=0).min())
# print(z.std(dim=0).max(),z.std(dim=0).min())
# gaussian = torch.distributions.Normal(
#     loc=z.mean(dim=0),
#     scale=z.std(dim=0)
# )
# zp = gaussian.sample([100])
perturb_x = real_x + torch.randn_like(real_x) * 0.01
loss_real = model.get_loss(real_x)
loss_preturb = model.get_loss(perturb_x)
print(loss_real.mean(),loss_preturb.mean())
# # save images
# grid = torchvision.utils.make_grid(x.reshape(-1,1,28,28).cpu(), nrow=10)
# torchvision.utils.save_image(grid, f'./play/epoch_{epoch}.png')