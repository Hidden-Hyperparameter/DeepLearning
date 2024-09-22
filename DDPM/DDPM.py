import torch
import torch.nn as nn
import torch.nn.functional as F

class SinousEmbedding(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        assert dim%2==0,NotImplementedError()
        self.angles = (10000.**(-2/dim))**torch.arange(1,dim//2+1,1,dtype=torch.float).cuda()
        self.angles.requires_grad_(False)
    def forward(self,x):
        angles = torch.einsum('m,i->im',self.angles,x.float())
        return torch.cat((torch.sin(angles),torch.cos(angles)),dim=1)
    
class ResidualBlock(nn.Module):

    def __init__(self,channels=128,kernel_size=3,t_dim=64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=kernel_size,padding=kernel_size//2)
        self.t_net = nn.Linear(t_dim,channels)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=kernel_size,padding=kernel_size//2)
        self.conv2.weight.data.fill_(0)
        self.conv2.bias.data.fill_(0)
    
    def forward(self,x,t):
        xc = x.clone()
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.conv1(x.relu())
        x = x + self.t_net(t).unsqueeze(-1).unsqueeze(-1).expand(t.shape[0],x.shape[1],x.shape[2],x.shape[3])
        x = self.conv2(x.relu())
        return x + xc

class Attention(nn.Module):

    def __init__(self,channels=128,attn_dim=32) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.Q = nn.Conv2d(channels,attn_dim,kernel_size=1,bias=False)
        self.K = nn.Conv2d(channels,attn_dim,kernel_size=1,bias=False)
        self.V = nn.Conv2d(channels,channels,kernel_size=1,bias=False)
        self.out_proj = nn.Conv2d(channels,channels,kernel_size=1)
        self.Q.weight.data.normal_(0,0.02)
        self.K.weight.data.normal_(0,0.02)
        self.V.weight.data.normal_(0,0.02)
    
    def forward(self,x):
        xc = x.clone()
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        attn_score = torch.einsum('bchw,bcxy->bhwxy',q,k).reshape(q.shape[0],*q.shape[-2:],-1)
        attn_score = attn_score.softmax(dim=-1).reshape(q.shape[0],*q.shape[-2:],*k.shape[-2:])
        return self.out_proj(torch.einsum('bhwxy,bcxy->bchw',attn_score,v)) + xc

class F_x_t(nn.Module):

    def __init__(self,in_channels,out_channels,out_size,kernel_size=3,t_shape=64,attn=False,residual=True) -> None:
        super().__init__()
        # self.t_channels = out_channels // 2
        # self.conv_channels = out_channels - self.t_channels
        self.t_channels = out_channels
        self.conv_channels = out_channels
        self.conv = nn.Conv2d(in_channels, self.conv_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.out_size = out_size
        self.fc = nn.Linear(t_shape, self.t_channels)
        self.attn = attn
        self.residual = residual
        if attn:
            self.attentions = nn.ModuleList([Attention(channels=self.conv_channels,attn_dim=out_channels) for _ in range(2)])
        if residual:
            self.ress = nn.ModuleList([ResidualBlock(channels=out_channels,kernel_size=kernel_size,t_dim=t_shape) for _ in range(2)])
        # self.fc = nn.Embedding(t_shape, self.t_num)

    def forward(self, x, t):
        if self.t_channels == 0:
            raise NotImplementedError()
            return self.conv(x)
        # return torch.cat([self.conv(x),self.fc(t).unsqueeze(-1).unsqueeze(-1).expand(t.shape[0], self.t_channels, self.out_size, self.out_size)],dim=1).relu()
        val = self.conv(x) + self.fc(t).unsqueeze(-1).unsqueeze(-1).expand(t.shape[0], self.t_channels, self.out_size, self.out_size)
        if self.residual:
            for i,ly in enumerate(self.ress):
                val = ly(val,t)
                if self.attn:
                    val = self.attentions[i](val)
        return val

class DDPM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.t_embedding_dim = 32
        self.t_embedding = SinousEmbedding(dim=self.t_embedding_dim)
        self.up= nn.ModuleList([
            F_x_t(in_channels=1,out_channels=32,out_size=32,kernel_size=3,t_shape=self.t_embedding_dim),
            F_x_t(in_channels=32,out_channels=64,out_size=16,kernel_size=3,t_shape=self.t_embedding_dim,attn=True),
            F_x_t(in_channels=64,out_channels=128,out_size=8,kernel_size=3,t_shape=self.t_embedding_dim,attn=True),
            # ResidualBlock(channels=128,kernel_size=3,t_dim=self.t_embedding_dim),
            # F_x_t(in_channels=128,out_channels=128,out_size=4,kernel_size=1,t_shape=self.t_embedding_dim),
        ])
        self.middle = nn.ModuleList([
            # nn.Identity()
            F_x_t(in_channels=128,out_channels=128,out_size=4,kernel_size=3,t_shape=self.t_embedding_dim,attn=True),
            # ResidualBlock(channels=128,kernel_size=3,t_dim=self.t_embedding_dim),
            # F_x_t(in_channels=128,out_channels=128,out_size=4,kernel_size=1,t_shape=self.t_embedding_dim,attn=False),
        ])
        self.down= nn.ModuleList([
            # F_x_t(in_channels=128,out_channels=128,out_size=2,kernel_size=1,t_shape=self.t_embedding_dim),
            F_x_t(in_channels=128,out_channels=64,out_size=8,kernel_size=3,t_shape=self.t_embedding_dim,attn=True),
            F_x_t(in_channels=64,out_channels=32,out_size=16,kernel_size=3,t_shape=self.t_embedding_dim,attn=True),
            F_x_t(in_channels=32,out_channels=16,out_size=32,kernel_size=3,t_shape=self.t_embedding_dim),
        ])
        # self.end_mlp = nn.Conv2d(32,1,kernel_size=3,padding=1)
        self.end_mlp = nn.Conv2d(16,1,kernel_size=1)

    def forward(self,x,t):
        x = x.reshape(-1,1,28,28)
        x = F.pad(x,(2,2,2,2),mode='constant',value=0)
        ttensor = self.t_embedding(t) # [batch, 256]
        batch = x.shape[0]
        # xc = x.clone()            print(attn_score.shape)

        ups = []
        for i,ly in enumerate(self.up):
            x = ly(x,ttensor)
            ups.append(x.clone()) # append: 28x28, 14x14
            x = nn.AvgPool2d(2)(x)
            # print('up,',i);print_range(x)
        for i,ly in enumerate(self.middle):
            # x = ly(x,ttensor)
            x = ly(x,ttensor)
            # print('middle,',i);print_range(x)
        for i,ly in enumerate(self.down):
            x = nn.Upsample(scale_factor=2)(x) + ups.pop() # 14x14, 28x28
            x = ly(x,ttensor)
            # x = nn.Upsample(scale_factor=2)(x) + ups.pop()
            # print('down,',i);print_range(x)
        x = self.end_mlp(x)
        x = x[:,:,2:30,2:30]
        return x.reshape(batch,28*28)