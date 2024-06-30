import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

class MaskedConv2d(nn.Conv2d):

    @staticmethod
    def make_mask(masktype,kernel_size,in_channels,out_channels):
        """
        mask: 1 if inputs is visible
        mask A: center is not visible
        mask B: center is visible
        """
        assert masktype in 'AB'
        assert kernel_size%2==1
        twod_size = kernel_size*kernel_size
        twod = torch.zeros(twod_size)
        twod[:twod_size//2] = 1
        if masktype == 'B':
            twod[twod_size//2] = 1
        twod = twod.reshape(kernel_size,kernel_size)
        return twod.expand(1,in_channels,kernel_size,kernel_size).to(device)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size:int,
        masktype:str,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding)
        self.mask = self.make_mask(masktype,kernel_size=kernel_size,in_channels=in_channels,out_channels=out_channels)

    def forward(self,x):
        self.weight.data *= self.mask
        return super().forward(x)

class MaskedConvLayer(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,masktype):
        super().__init__()
        self.maskedconv = MaskedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            masktype=masktype,
            padding=kernel_size//2,
        )
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        x = self.maskedconv(x)
        return self.bn(x)

class ResidualLayer(nn.Module):

    def __init__(self,channels,kernel_size,act=torch.relu):
        super().__init__()
        self.conv1 = MaskedConvLayer(channels,channels,kernel_size,'B')
        self.conv2 = MaskedConvLayer(channels,channels,kernel_size,'B')
        self.act = act

    def forward(self,x):
        y = self.conv1(x)
        y = torch.relu(y)
        y = self.conv2(y)
        return self.act(x+y)


class PixelCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            MaskedConv2d(1,128,7,'A',padding=3),
        ] + [
            ResidualLayer(
                channels=128,kernel_size=5
            ) for _ in range(5)
        ]
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=192,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=192,out_channels=256,kernel_size=1),
        )

    def forward(self,x):
        for layer in self.convs:
            x = layer(x)
            if not isinstance(layer,ResidualLayer):
                x = torch.relu(x)
        return self.out_proj(x)

    def get_loss(self,x):
        """negative log likelihood"""
        logits = self(x).permute(0,2,3,1).reshape(-1,256) # shape: (batch*28*28,256)
        # quantize x to 0-255 discrete values
        x_quantized = torch.floor(x.clamp(1e-5,1-1e-5)*256).long() # shape: (batch,1,28,28)
        # print('x_quantized:',x_quantized.min(),x_quantized.max())
        x_quantized = x_quantized.permute(0,2,3,1).reshape(-1) # shape: (batch*28*28)
        loss = F.cross_entropy(logits,x_quantized) * (28 * 28)
        return loss

    @torch.no_grad()
    def sample(self,batch=1):
        out = torch.zeros(batch,1,28,28).to(device)
        for i in tqdm.trange(28,desc='Generation'):
            for j in range(28):
                logits = self(out)[...,i,j] # shape: (batch,256)
                distr = torch.distributions.Categorical(logits=logits)
                out[:,0,i,j] = distr.sample().flatten().float()/256
        return out

