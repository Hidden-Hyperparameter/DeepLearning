import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

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
        return twod.expand(1,in_channels,kernel_size,kernel_size)

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


class PixelCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            MaskedConv2d(1,128,7,'A',padding=3),
            MaskedConv2d(128,64,5,'B',padding=2),
        ] + [
            MaskedConv2d(64,64,5,'B',padding=2),
        ] * 5
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=1,kernel_size=1),
        )

    def forward(self,x):
        for layer in self.convs:
            x = layer(x)
            x = torch.relu(x)
        return self.out_proj(x)

    def get_loss(self,x):
        """negative log likelihood"""
        out = torch.sigmoid(self(x)) # shape: (batch,1,28,28)
        return -torch.log(out).sum(dim=[1,2,3]).mean(dim=0)

    @torch.no_grad()
    def sample(self,batch=1):
        out = torch.zeros(batch,1,28,28)
        for i in tqdm.trange(28):
            for j in range(28):
                logits = self(out)[:,0,i,j] # shape: (batch)
                distr = torch.distributions.Bernoulli(logits=logits)
                out[:,0,i,j] = distr.sample().flatten()
        return out