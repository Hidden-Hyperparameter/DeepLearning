EP = 49
from model import VQVAE
# from pixel import PixelCNN,device
from lstm import LSTMModel,device
import torch,os
vqvae = VQVAE().to(device)
vqvae.load_state_dict(torch.load(os.path.join('models',f'vqvae_ep{EP:02d}.pth')))
vqvae.eval()
# pixel = PixelCNN(num_class=vqvae.num_embeddings).to(device)
# pixel.load_state_dict(torch.load(os.path.join('models',f'pixelcnn_ep{EP:02d}.pth')))
# pixel.eval()

lstm = LSTMModel(num_classes=vqvae.num_embeddings).to(device)
lstm.load_state_dict(torch.load(os.path.join('models',f'lstm_ep{EP:02d}.pth')))
lstm.eval()

z_s = lstm.generate().reshape(-1,vqvae.latent_size,vqvae.latent_size)
images = vqvae.generate(z_s).cpu() # [batch,1,64,64]

import torchvision.utils as vutils
vutils.save_image(images, os.path.join('samples',f'sample{EP:02d}.png'))
print('\n\ndone')