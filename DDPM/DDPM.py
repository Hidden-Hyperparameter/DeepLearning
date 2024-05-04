import torch
import torch.nn as nn
import torch.nn.functional as F
class DDPM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_size = 28 * 28
        self.t_net = nn.Linear(1001,100)
        self.conv1 = nn.Sequential(
             nn.Conv2d(1,64,kernel_size=(3,3)),
             nn.BatchNorm2d(64),
             nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(44265,self.in_size)
        )
    def forward(self,x,t):
        x = x.reshape(-1,1,28,28)
        ttensor = F.one_hot(t,num_classes=1001)
        batch = x.shape[0]
        x =  self.conv1(x)
        x = x.reshape(batch,-1)
        # print(x.shape)
        # print(ttensor.shape)
        cat = torch.cat((x,ttensor),dim=1)
        return self.out(cat)