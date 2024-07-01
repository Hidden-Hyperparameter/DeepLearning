import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from typing import Literal

from pixelcnn import device,MaskedConv2d

class VerticalConv(MaskedConv2d):

    @staticmethod
    def make_mask(masktype:Literal['A','B'],kernel_size,in_channels,out_channels):
        # print('overloading make_mask successful')
        assert kernel_size%2==1
        twod = torch.zeros(kernel_size,kernel_size)
        twod[:(kernel_size//2)+1,:] = 1
        return twod.expand(1,in_channels,kernel_size,kernel_size).to(device)
    
class HorizontalConv(MaskedConv2d):

    @staticmethod
    def make_mask(masktype:Literal['A','B'],kernel_size,in_channels,out_channels):
        # print('overloading make_mask successful')
        assert masktype in 'AB'
        assert kernel_size%2==1
        twod = torch.zeros(kernel_size,kernel_size)
        if masktype == 'A':
            twod[(kernel_size-1)//2,:(kernel_size-1)//2] = 1
        else:
            twod[(kernel_size-1)//2,:(kernel_size+1)//2] = 1
        return twod.expand(1,in_channels,kernel_size,kernel_size).to(device)
    
class GatedConvLayer(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,convtype:Literal['H','V'],masktype:Literal['A','B']):
        super().__init__()
        assert convtype in 'HV' and masktype in 'AB'
        if convtype == 'H':
            self.maskedconv = HorizontalConv(
                in_channels=in_channels,
                out_channels=out_channels*2,
                kernel_size=kernel_size,
                masktype=masktype,
                padding=kernel_size//2,
            )
        else:
            self.maskedconv = VerticalConv(
                in_channels=in_channels,
                out_channels=out_channels*2,
                kernel_size=kernel_size,
                masktype=masktype,
                padding=kernel_size//2,
            )
        self.out_channels = out_channels
        self.bn = nn.BatchNorm2d(out_channels*2)

        self.apply_res = False # see note.md, residual connection isn't very useful.
        # self.apply_res = not(masktype == 'A' and convtype == 'H')

        self.linear = nn.Linear(10,out_channels*2) # as mentioned in the paper, the encode of label h doesn't specify "where" to have the feature, so this is independent of pixel position.

    def forward(self,x,h):
        """perform conditioal generation with h being the one-hot label information"""
        batch = x.shape[0]
        xc = x.clone()
        x = self.maskedconv(x)
        x = self.bn(x)
        h_embedded = self.linear(h)
        x_first = x[:,:self.out_channels] # (batch, out_channel, 28, 28)
        x_second = x[:,self.out_channels:]
        h_first = h_embedded[:,:self.out_channels] # (batch, out_channel)
        h_second = h_embedded[:,self.out_channels:]
        out = torch.tanh(x_first + h_first.reshape(batch,self.out_channels,1,1)) * torch.sigmoid(x_second + h_second.reshape(batch,self.out_channels,1,1)) # gated activation

        # if not type A, can have residual connection'
        if self.apply_res:
            return out + xc
        else:
            return out
    
class GatedPixelCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.vertical_stack = nn.ModuleList([
            GatedConvLayer(in_channels=1,out_channels=64,kernel_size=7,convtype='V',masktype='B'),
        ] + [
            GatedConvLayer(in_channels=64,out_channels=64,kernel_size=7,convtype='V',masktype='B') for _ in range(5)
        ])
        self.horizontal_stack = nn.ModuleList([
            GatedConvLayer(in_channels=64,out_channels=64,kernel_size=7,convtype='H',masktype='A'),
        ] + [
            GatedConvLayer(in_channels=64,out_channels=64,kernel_size=7,convtype='H',masktype='B') for _ in range(5)
        ])
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=1),
        )
    
    def forward(self,x,h):
        h = F.one_hot(h.to(torch.long),num_classes=10).float()
        y = x.clone()
        ys = []
        for layer in self.vertical_stack:
            y = layer(y,h)
            # add the shift-downward image into the list
            ys.append(torch.cat(
                (torch.zeros(y.shape[0],y.shape[1],1,28).to(device),y[...,:-1,:]),dim=2
            ))
        for i,layer in enumerate(self.horizontal_stack):
            x = x + ys[i]
            x = layer(x,h)
        x = self.out_proj(x)
        return x
    
    def get_loss(self,x,h):
        """negative log likelihood"""
        logits = self(x,h).permute(0,2,3,1).reshape(-1,256) # shape: (batch*28*28,256)
        # quantize x to 0-255 discrete values
        x_quantized = torch.floor(x.clamp(1e-5,1-1e-5)*256).long() # shape: (batch,1,28,28)
        # print('x_quantized:',x_quantized.min(),x_quantized.max())
        x_quantized = x_quantized.permute(0,2,3,1).reshape(-1) # shape: (batch*28*28)
        loss = F.cross_entropy(logits,x_quantized) * (28 * 28)
        return loss

    @torch.no_grad()
    def sample(self,batch=1):
        self.eval()
        out = torch.zeros(batch,1,28,28).to(device)
        label = torch.arange(10).repeat(batch//10)[:batch].to(device)
        for i in tqdm.trange(28,desc='Generation'):
            for j in range(28):
                logits = self(out,label)[...,i,j] # shape: (batch,256)
                distr = torch.distributions.Categorical(logits=logits)
                out[:,0,i,j] = distr.sample().flatten().float()/256
        return out