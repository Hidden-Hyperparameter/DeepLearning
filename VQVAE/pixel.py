import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MaskedConv2d(nn.Conv2d):
    """
    mask A:
        1 1 1
        1 1 0
        0 0 0

    mask B:
        1 1 1
        1 0 0
        0 0 0
    """

    @staticmethod
    def make_mask(in_channels: int, out_channels: int, kernel_size: int, mask_type: Literal['A','B']) -> torch.Tensor:
        assert kernel_size % 2 == 1, 'kernel_size must be odd'
        if mask_type not in 'AB':
            raise ValueError('mask_type must be either A or B')
        mask = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
        mask[:,:,:kernel_size//2,:] = 1
        mask[:, :, kernel_size // 2, :kernel_size // 2] = 1
        if mask_type == 'A':
            mask[:, :, kernel_size // 2, kernel_size // 2] = 1
        return mask.float()

    def __init__(self, mask_type:Literal['A','B'], in_channels: int, out_channels: int, kernel_size: int, stride: int | torch.Tuple[int] = 1, padding: str | int | torch.Tuple[int] = 0, dilation: int | torch.Tuple[int] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.mask = MaskedConv2d.make_mask(in_channels, out_channels, kernel_size, mask_type).to(device)
    
    def forward(self, x):
        self.weight.data.mul_(self.mask)
        return super().forward(x)

class PixelCNN(nn.Module):

    def __init__(self,num_class):
        super().__init__()
        self.convs = nn.Sequential(
            MaskedConv2d('B',1,128,3,padding=1),
            nn.ReLU(),
            MaskedConv2d('A',128,128,3,padding=1),
            nn.ReLU(),
            MaskedConv2d('A',128,128,3,padding=1),
            nn.ReLU(),
            MaskedConv2d('A',128,128,3,padding=1),
            nn.ReLU(),
        )
        self.num_classes = num_class
        self.output = nn.Conv2d(128,self.num_classes,1)

    def get_loss(self,x):
        # x.shape: [batch,9,9]
        x = x.to(torch.long)
        assert x.max() < self.num_classes and x.min() >= 0, f'Invalid data, having x.max(): {x.max()}, x.min(): {x.min()}'
        out = self.output(self.convs(x.unsqueeze(1) / self.num_classes)) # [batch,128,9,9]
        return F.cross_entropy(out.permute(0,2,3,1).reshape(-1,self.num_classes),x.reshape(-1))

    @torch.no_grad()
    def generate(self,num:int=100):
        init = torch.zeros(num,1,9,9).to(torch.long).to(device)
        for i in range(9):
            for j in range(9):
                out = self.output(self.convs(init / self.num_classes))
                probs = F.softmax(out[:,:,i,j],dim=-1) # [batch,512]
                init[:,0,i,j] = torch.distributions.Categorical(probs).sample()
        return init.squeeze(1)