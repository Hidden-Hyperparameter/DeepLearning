import torch
import torch.nn as nn
import torch.nn.functional as F

class Res(nn.Module):
    def __init__(self, channel, kernel_size=3,x_size = 28) -> None:
        super().__init__()
        self.channel = channel
        self.conv1=nn.Sequential(
            nn.Conv2d(channel,channel,(kernel_size,kernel_size),padding=(kernel_size-1)//2),
            nn.BatchNorm2d(channel),
        )
        self.conv2= nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channel,channel,(kernel_size,kernel_size),padding=(kernel_size-1)//2),
        )
        self.t_net = nn.Linear(256,self.channel*x_size*x_size)
    def forward(self,x,t):
        res = x.clone()
        x = self.conv1(x)
        x = x + self.t_net(t).reshape(x.shape)
        x = self.conv2(x) + res
        return x

class SinousEmbedding(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        assert dim%2==0,NotImplementedError()
        self.angles = (10000**(-2/dim))**torch.arange(1,dim//2+1,1).cuda()
        self.angles.requires_grad_(False)
    def forward(self,x):
        angles = torch.einsum('m,i->im',self.angles,x)
        return torch.cat((torch.sin(angles),torch.cos(angles)),dim=1)

class Attention(nn.Module):
    def __init__(self, channel,hidden_size=512) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(channel,hidden_size)
        self.k_proj = nn.Linear(channel,hidden_size)
        self.v_proj = nn.Linear(channel,hidden_size)
        self.out_proj = nn.Linear(hidden_size,channel)
    def forward(self,x):
        res = x.clone()
        batch,channel = x.shape[:2]
        seq_len = x.shape[-1]
        x = x.reshape(batch,channel,seq_len*seq_len).transpose(1,2)
        v = self.v_proj(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        att_sc = torch.einsum('bic,bjc->bij',q,k)*((self.hidden_size)**-0.5)
        att_sc = torch.softmax(att_sc,dim=-1)
        att_out = torch.einsum('bij,bjc->bic',att_sc,v)
        ans = self.out_proj(att_out).transpose(1,2).reshape(batch,channel,seq_len,seq_len)
        return ans+res


class ResBlockWithAttention(nn.Module):
    def __init__(self, in_channel, out_channel,x_size,with_attention=True,kernel_size=3) -> None:
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,(kernel_size,kernel_size),padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.reses = nn.ModuleList(
            [Res(out_channel,kernel_size,x_size) for _ in range(4)]
        )
        if with_attention:
            self.attentions = nn.ModuleList(
                [Attention(out_channel) for _ in range(4)]
            )
        
    def forward(self,x,t):
        x = self.conv(x)
        for i,ly in enumerate(self.reses):
            x = ly(x,t)
            if hasattr(self,'attentions'):
                x = self.attentions[i](x)
        return x


class DDPM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_size = 28 * 28
        self.t_embedding = SinousEmbedding(dim=256)
        self.up= nn.ModuleList([
            ResBlockWithAttention(1,64,x_size=28,with_attention=False), # 28 28
            nn.MaxPool2d(kernel_size=(2,2)), # 14 14
            ResBlockWithAttention(64,128,x_size=14), # 14 14
            nn.MaxPool2d(kernel_size=(2,2)), # 7 7
            ResBlockWithAttention(128,256,x_size=7), # 7 7
        ])
        self.middle = nn.ModuleList([
            nn.Conv2d(256,256,kernel_size=(5,5),padding=2),
            nn.ReLU(),
            Attention(256)
        ])
        self.down= nn.ModuleList([
            ResBlockWithAttention(512,128,x_size=7), # 7 7
            nn.Upsample(scale_factor=2),
            ResBlockWithAttention(256,64,x_size=14),
            nn.Upsample(scale_factor=2),
            ResBlockWithAttention(128,1,x_size=28,with_attention=False),       
        ])

    def forward(self,x,t):
        x = x.reshape(-1,1,28,28)
        ttensor = self.t_embedding(t)
        batch = x.shape[0]
        ups = []
        for i,ly in enumerate(self.up):
            if isinstance(ly,ResBlockWithAttention):
                x = ly(x,ttensor)
                cl = x.clone()
                ups.append(cl)
            else:
                x = ly(x)
        for ly in self.middle:
            x = ly(x)
        for ly in self.down:
            if isinstance(ly,ResBlockWithAttention):
                old = ups.pop()
                x = ly(torch.cat((x,old),dim=1),ttensor)
            else:
                x = ly(x)
        x = x.reshape(batch,-1)
        return x