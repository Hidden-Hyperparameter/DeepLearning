import torch
import torch.nn as nn
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
float_tp = torch.float32
torch.autograd.anomaly_mode.set_detect_anomaly(True)
from math import log

def detect(x:torch.Tensor):
    assert not torch.isnan(x).any(),x
    print(x)

class Coupling(nn.Module):

    @staticmethod
    def make_alternating_mask(size:int,masktype:str) -> torch.Tensor:
        """
        make a alternating mask like chess board. 

        Output Shape: [size,size]
        """
        assert masktype in 'AB',NotImplementedError()
        mask = torch.ones(size*size)
        if masktype == 'A':
            mask[::2] = 0
        else:
            mask[1::2] = 0
        return mask.reshape(size,size)

    def __init__(self, masktype:str,size:int,channel,kernel_size=3) -> None:
        super().__init__()
        self.masktype = masktype
        self.mask = Coupling.make_alternating_mask(size,masktype).expand(1,channel,size,size).to(device)
        self.mask.requires_grad_(False)
        self.drop = 0
        self.alpha_net = nn.Sequential(
            nn.Conv2d(channel,128,kernel_size,padding=kernel_size//2),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(self.drop),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size,padding=kernel_size//2),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(self.drop),
            nn.ReLU(),
            nn.Conv2d(128,channel,kernel_size,padding=kernel_size//2),
            nn.BatchNorm2d(channel),
            # nn.Tanh()
        )
        self.mu_net = nn.Sequential(
            nn.Conv2d(channel,128,kernel_size,padding=kernel_size//2),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(self.drop),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size,padding=kernel_size//2),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(self.drop),
            nn.ReLU(),
            nn.Conv2d(128,channel,kernel_size,padding=kernel_size//2),
            nn.BatchNorm2d(channel),
        )
        # init parameters
        for m in self.parameters():
            m.data.normal_(std=0.01)


    def forward(self,x):
        """
        Calculate z = f(x) using mask
        x: batched input        
        z = mask * x + (1-mask) * [x * exp(alpha(mask*x)) + mu(mask*x)]
        """
        alpha = self.alpha_net(self.mask * x)
        ans = self.mask * x + x * torch.exp((1-self.mask) * alpha)*(1-self.mask) + self.mu_net(self.mask * x) * (1-self.mask)
        # print('alpha:',alpha*(1-self.mask))
        logdet = torch.sum(alpha*(1-self.mask),dim=[1,2,3])

        return ans,logdet
    
    def backward(self,z):
        """
        Used in generation, use z to calculate x
        """
        ans = z * self.mask + (1-self.mask) * (z * (1-self.mask) - self.mu_net(self.mask * z)) * torch.exp(-self.alpha_net(self.mask * z))
        return ans
    
class Squeeze(nn.Module):

    @staticmethod
    def forward(x):
        """
        Squeeze x, from [c,size,size] to [4*c,size/2,size/2]
        x: batched input
        """
        # print('before squeeze',x.shape)
        assert x.shape[-1] % 2 == 0, NotImplementedError()
        x00 = x[...,::2,::2]
        x01 = x[...,::2,1::2]
        x10 = x[...,1::2,::2]
        x11 = x[...,1::2,1::2]
        out = torch.cat([x00,x01,x10,x11],dim=1)
        return out,0
    
    @staticmethod
    def backward(z):
        x = torch.zeros(z.shape[0],z.shape[1]//4,z.shape[2]*2,z.shape[3]*2).to(device).to(float_tp)
        x[...,::2,::2] = z[:,:(z.shape[1]//4),...]
        x[...,::2,1::2] = z[:,(z.shape[1]//4):(z.shape[1]//2),...]
        x[...,1::2,::2] = z[:,(z.shape[1]//2):(z.shape[1]//4)*3,...]
        x[...,1::2,1::2] = z[:,(z.shape[1]//4)*3:,...]
        return x
    
class Flow(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.channel = 1

        self.layer0 = nn.ModuleList([
            Coupling(masktype='A',size=28,channel=self.channel),
            Coupling(masktype='B',size=28,channel=self.channel),
            # Coupling(masktype='A',size=28,channel=self.channel),
            Squeeze(),
            # Coupling(masktype='B',size=14,channel=4 * self.channel),
            Coupling(masktype='A',size=14,channel=4 * self.channel),
            Coupling(masktype='B',size=14,channel=4 * self.channel),
        ])

        self.layer1 = nn.ModuleList([
            Coupling(masktype='A',size=14,channel=2 * self.channel),
            Coupling(masktype='B',size=14,channel=2 * self.channel),
            # Coupling(masktype='A',size=14,channel=2 * self.channel),
            Squeeze(),
            # Coupling(masktype='B',size=7,channel=8 * self.channel),
            Coupling(masktype='A',size=7,channel=8 * self.channel),
            Coupling(masktype='B',size=7,channel=8 * self.channel),
        ])

        self.layer2 = nn.ModuleList([
            Coupling(masktype='A',size=7,channel=4 * self.channel),
            Coupling(masktype='B',size=7,channel=4 * self.channel),
            # Coupling(masktype='A',size=7,channel=4 * self.channel),
        ])

        self.layers = [self.layer0,self.layer1,self.layer2]

    def forward(self,x):
        batch = x.shape[0]
        logdet = 0
        zs = []
        for i,layeri in enumerate(self.layers):
            # print('----layer {}----'.format(i))
            for layer in layeri:
                # print('-----{} layer----'.format(type(layer)))
                x,one = layer(x)
                # print(x)
                # print('logdet contribute',one)            
                logdet += one
            if i != len(self.layers)-1:
                channel_dim = x.shape[1]
                z = x[:,:(channel_dim//2),...]
                x = x[:,(channel_dim//2):,...]
                zs.append(z.clone())
        zs.append(x)
        latent = torch.cat([z.reshape(batch,-1) for z in zs],dim=-1) # isomorphic Gaussian
        return latent,logdet
    
    def backward(self,z):
        # z0: 14 * 14 * (2*self.channel), z1: 7 * 7 * (4*self.channel), x: 7 * 7 * (4*self.channel)
        l1 = 14 * 14 * (2*self.channel)
        l2 = 7 * 7 * (4*self.channel)
        zs = [
            z[:,l1+l2:].reshape(-1,4*self.channel,7,7), # z2
            z[:,l1:l1+l2].reshape(-1,4*self.channel,7,7), # z1
            z[:,:l1].reshape(-1,2*self.channel,14,14), # z0
        ]
        z = zs[0]
        for i,layeri in enumerate(reversed(self.layers)):
            if i != 0:
                z = torch.cat([zs[i],z],dim=1)
            # print(f'layer {i}',z.shape)
            for layer in reversed(layeri):
                z = layer.backward(z)

        return z
    
    def get_loss(self,x):

        latent,logdet = self(x)
        loss =  (torch.sum(latent**2,dim=-1) / 2 - logdet).mean(dim=0)
        loss += 784 * self.channel * 0.5 * log(torch.tensor([2*3.1415,]).to(device))
        # if loss < 0:
        #     print('latent loss:',torch.sum(latent**2,dim=-1)[0].item())
        #     print('logdet loss:',logdet[0].item())
        return loss