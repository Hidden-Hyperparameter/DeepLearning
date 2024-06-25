import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torchvision

class Decoder(nn.Module):

    def __init__(self,latent_size) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.begin_size = 12
        self.begin = nn.Sequential(
            nn.Linear(self.latent_size+50,(self.begin_size*self.begin_size)*64),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=128,kernel_size=4,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=4,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,out_channels=1,kernel_size=5,padding=2,stride=2),
            nn.ReLU()
        )

    def forward(self,z,y):
        y = F.one_hot(y,num_classes=10).to(y.device)
        y = y.unsqueeze(-1).expand([y.shape[0],10,5]).reshape(y.shape[0],-1)
        # print(z.shape,y.shape)
        m = torch.cat((z,y),dim=-1)
        m = self.begin(m).reshape(-1,64,self.begin_size,self.begin_size)
        m = self.net(m).reshape(-1,29,29)
        return torchvision.transforms.CenterCrop([28,28])(m)

class Encoder(nn.Module):

    def __init__(self,latent_size) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.feature_net = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=8,kernel_size=5),
            nn.ReLU()
        )
        out_dim = 442
        self.mu_net = nn.Sequential(
            nn.Linear(out_dim,self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size,self.latent_size)
        )
        self.sigma_net = nn.Sequential(
            nn.Linear(out_dim,self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size,self.latent_size)
        )
        # self.y_net = nn.Linear(out_dim,10)

    def forward(self,x,y):
        batch = x.shape[0]
        features = self.feature_net(x).reshape(batch,-1)
        label_onehot = F.one_hot(y,num_classes=10).to(y.device)
        label_onehot = label_onehot.unsqueeze(-1).expand([batch,10,5]).reshape(batch,-1)
        features = torch.cat((features,label_onehot),dim=-1)
        z = (
            self.mu_net(features),
            self.sigma_net(features) # logstd!
        )
        # y = self.y_net(features)
        return z

class VAE(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.latent_size = 100
        self.decoder = Decoder(latent_size=self.latent_size)
        self.encoder = Encoder(latent_size=self.latent_size)

        self.x_var = 1

        self.bkw_cnt = 0

    def get_loss(self,x,y):
        (mean,logstd) = self.encoder(x,y)
        var = torch.exp(2*logstd)
        # self.bkw_cnt += 1
        # if self.bkw_cnt % 100 == 0:
        #     print(torch.max(mean),torch.min(mean))
        #     print(torch.max(logstd),torch.min(logstd))
        kl = 0.5 * (torch.sum(var,dim=-1) + torch.sum(mean**2,dim=-1) - 2 * torch.sum(logstd,dim=-1))
        sample = Normal(loc=mean,scale=var**0.5).rsample([1]).squeeze(0).to(x.device)
        reconstruct = torch.sum((self.decoder(sample,y)-x.squeeze(1))**2,dim=[1,2])/(2*self.x_var)
        # label_loss = F.cross_entropy(label,y)
        return reconstruct.mean(dim=0),kl.mean(dim=0)#,label_loss