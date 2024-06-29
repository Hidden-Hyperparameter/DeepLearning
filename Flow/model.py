import torch
import torch.nn as nn
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
float_tp = torch.float32
import sys,os
sys.path.append(os.path.abspath('..'))
torch.autograd.anomaly_mode.set_detect_anomaly(True)
from math import log
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

def detect(x:torch.Tensor):
    assert not torch.isnan(x).any(),x
    assert not torch.isinf(x).any(),x
    # print(x)
    
class Res(nn.Module):

    def __init__(self,size,channel,kernel_size=3,activation=torch.relu,with_mlp=False) -> None:
        super().__init__()
        self.activation = activation
        std_list = [
            nn.Conv2d(channel,128,kernel_size=kernel_size,padding=kernel_size//2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(128,128,kernel_size=kernel_size,padding=kernel_size//2),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.Conv2d(128,channel,kernel_size=kernel_size,padding=kernel_size//2),
            nn.BatchNorm2d(channel)
        ]
        if with_mlp:
            std_list += [
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(channel*size*size,256),
                nn.ReLU(),
                nn.Linear(256,channel*size*size),
            ]

        self.seq = nn.Sequential(*std_list)

    def forward(self,x):
        xc = x.clone()
        result = xc + self.seq(x).reshape_as(xc)
        if self.activation is not None:
            return self.activation(result)
        return result


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
            Res(size=size,channel=channel,kernel_size=kernel_size),
            Res(size=size,channel=channel,kernel_size=kernel_size,activation=torch.tanh,with_mlp=True),
        )
        self.mu_net = nn.Sequential(
            Res(size=size,channel=channel,kernel_size=kernel_size),
            Res(size=size,channel=channel,kernel_size=kernel_size,activation=None,with_mlp=True),
        )

    def forward(self,x):
        """
        Calculate z = f(x) using mask
        x: batched input        
        z = mask * x + (1-mask) * [x * exp(alpha(mask*x)) + mu(mask*x)]
        """
        # print(x.shape)
        alpha = torch.tanh(self.alpha_net(self.mask * x) + self.mask * x) # residual connection
        mu = self.mu_net(self.mask * x) + self.mask * x
        # print(alpha.shape)
        ans = self.mask * x + x * torch.exp((1-self.mask) * alpha)*(1-self.mask) + mu * (1-self.mask)
        # print('alpha:',alpha*(1-self.mask))
        logdet = torch.sum(alpha*(1-self.mask),dim=[1,2,3])
        # print(logdet.mean())
        # print((abs(ans)>3).float().mean())
        return ans,logdet
    
    def backward(self,z):
        """
        Used in generation, use z to calculate x
        """
        mu = self.mu_net(self.mask * z) + self.mask * z
        alpha = torch.tanh(self.alpha_net(self.mask * z) + self.mask * z)
        ans = z * self.mask + (1-self.mask) * (z * (1-self.mask) - mu) * torch.exp(-alpha)
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
        x = torch.zeros(z.shape[0],z.shape[1]//4,z.shape[2]*2,z.shape[3]*2).to(device)
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
            # BatchNorm(size=28,channel=self.channel),
            Squeeze(),
            Coupling(masktype='B',size=14,channel=4 * self.channel),
            Coupling(masktype='A',size=14,channel=4 * self.channel),
            # BatchNorm(size=14,channel=4*self.channel),
        ])
        self.prior = torch.distributions.Normal(torch.tensor([0.]).to(device),torch.tensor([1.]).to(device))

        self.layer1 = nn.ModuleList([
            Coupling(masktype='A',size=14,channel=2 * self.channel),
            Coupling(masktype='B',size=14,channel=2 * self.channel),
            # BatchNorm(size=14,channel=2*self.channel),
            Squeeze(),
            Coupling(masktype='B',size=7,channel=8 * self.channel),
            Coupling(masktype='A',size=7,channel=8 * self.channel),
            # BatchNorm(size=7,channel=8*self.channel),
        ])

        self.layer2 = nn.ModuleList([
            Coupling(masktype='A',size=7,channel=4 * self.channel),
            # BatchNorm(size=7,channel=4*self.channel),
            Coupling(masktype='B',size=7,channel=4 * self.channel),
            # BatchNorm(size=7,channel=4*self.channel),
        ])

        # self.layers = [self.layer0]
        self.layers = [self.layer0,self.layer1,self.layer2]

    def forward(self,x,pre_process=True):
        if pre_process:
            x = self.pre_process(x)
        batch = x.shape[0]
        logdet = 0
        zs = []
        for i,layeri in enumerate(self.layers):
            # print('----layer {}----'.format(i))
            for layer in layeri:
                x,one = layer(x)
                # if not self.training:
                #     print('After {} layer, statics:'.format(type(layer)),'min:',x.min().item(),'max:',x.max().item(),'mean:',x.mean().item(),'std:',x.std().item())
                # print('logdet shape:',one.shape)
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
    
    def backward(self,z,pre_process=True):
        # # z0: 14 * 14 * (2*self.channel), z1: 7 * 7 * (4*self.channel), x: 7 * 7 * (4*self.channel)
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
        if pre_process:
            return self.inv_preprocess(z) # inverse action of pre_process
        else:
            return z
        #### One Layer Version ####
        # z = z.reshape(-1,self.channel,28,28)
        # for layer in reversed(self.layer0):
        #     z = layer.backward(z)
        # return self.inv_preprocess(z)

    def pre_process(self,x):
        # return x
        return (x+1e-5).log() - (1-x+1e-5).log()

    def inv_preprocess(self,x):
        # return x
        return x.sigmoid()
    
    def get_loss(self,x,pre_process=True):
        latent,logdet = self(x,pre_process=pre_process)
        prior_logprob = self.prior.log_prob(latent).sum(dim=-1)
        loss = -(prior_logprob+logdet).mean(dim=0)
        return loss