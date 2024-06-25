import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class Decoder(nn.Module):

    def __init__(self,latent_size) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.begin_size = 12
        self.begin = nn.Linear(self.latent_size+10,(self.begin_size*self.begin_size)*64)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,padding=1),
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=4,padding=1),
            nn.ConvTranspose2d(in_channels=16,out_channels=8,kernel_size=4,padding=1,stride=2),
            nn.ConvTranspose2d(in_channels=8,out_channels=1,kernel_size=5,padding=1),
        )

    def forward(self,z,y):
        y = F.one_hot(y,num_classes=10)
        m = torch.cat((z,y),dim=-1)
        m = self.begin(m).reshape(-1,64,self.begin_size,self.begin_size)
        m = self.net(m)
        return m.reshape(-1,28,28)

class Encoder(nn.Module):

    def __init__(self,latent_size) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.feature_net = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=2),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3),
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=4,stride=2),
            nn.Conv2d(in_channels=16,out_channels=4,kernel_size=5),
        )
        out_dim = 196
        self.mu_net = nn.Linear(out_dim,self.latent_size)
        self.sigma_net = nn.Linear(out_dim,self.latent_size)
        self.y_net = nn.Linear(out_dim,10)

    def forward(self,x):
        batch = x.shape[0]
        features = self.feature_net(x)
        features = features.reshape(batch,-1)
        z = (
            self.mu_net(features),
            self.sigma_net(features) # logstd!
        )
        y = self.y_net(features)
        return z,y

class VAE(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.latent_size = 100
        self.decoder = Decoder(latent_size=self.latent_size)
        self.encoder = Encoder(latent_size=self.latent_size)

    def get_loss(self,x,y):
        (mean,logstd),label = self.encoder(x)
        std = torch.exp(2*logstd)
        kl = 0.5 * (torch.sum(std**2,dim=-1) + torch.sum(mean**2,dim=-1) - 2 * torch.sum(logstd,dim=-1))
        sample = Normal(loc=mean,scale=std**0.5).rsample([1]).squeeze(0)
        reconstruct = torch.sum((self.decoder(sample,y)-x)**2,dim=[1,2,3])
        label_loss = F.cross_entropy(label,y)
        loss = (reconstruct + kl).mean(dim=0) +  label_loss
        return loss