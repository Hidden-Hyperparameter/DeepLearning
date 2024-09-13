import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):

    def __init__(self,channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channel,channel,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.Conv2d(channel,channel,kernel_size=5,padding=2),
            # nn.ReLU(),
        )
    
    def forward(self,x):
        return x + self.conv(x)

class VQVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.latent_size = 2
        self.latent_channel = 4
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=5,stride=2,padding=2),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=5,stride=2,padding=2),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=5,stride=2,padding=2),
            nn.ReLU(),
            nn.Conv2d(64,self.latent_channel,kernel_size=5,stride=2,padding=2),
            # nn.Linear(784,256),
            # nn.ReLU(),
            # nn.Linear(256,64),
            # nn.ReLU(),
            # nn.Linear(64,self.latent_channel),
        )
        self.feature_conv = nn.Sequential(
            nn.ConvTranspose2d(self.latent_channel,128,kernel_size=5,stride=2,padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=5,stride=2,padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=5,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32,1,kernel_size=5,stride=2),
            # nn.Linear(self.latent_channel,784),
        )
        self.foo = nn.Parameter(torch.randn(1))


        self.num_embeddings = 64
        self.dictionary = nn.Embedding(self.num_embeddings,self.latent_channel) # dimension is 128
        self.BETA = .25
        self.RATE = 1
        self.ENCODER_RATE = 1
        # self.AVG_GAMMA = 0.99
        self.apply_init()

        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)

    def apply_init(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Embedding):
                nn.init.normal_(m.weight,std=1)
        pass

    def encode(self,x):
        # x = x.reshape(x.shape[0],-1)
        z = self.encoder(x)
        # print('after encoder,',z.shape)
        return z.reshape(z.shape[0],self.latent_channel,self.latent_size,self.latent_size)

    def decode(self,z):
        # z = z.reshape(z.shape[0],-1)
        z = self.feature_conv(z)
        # print(z.shape)
        z = z[...,:28,:28]
        return z

    @torch.no_grad()
    def nearest_neighbor(self,z:torch.Tensor):
        diff = z.unsqueeze(1) - self.dictionary.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [batch, self.num_embedding, 128, 9, 9]
        best = diff.norm(dim=2).argmin(dim=1) # [batch, 9, 9]
        return best

    def update(self,x,do_update=True):
        z = self.encode(x)
        # print('z.range:',z.min().item(),z.max().item())
        index = self.nearest_neighbor(z) # [batch, 9, 9]
        val = self.dictionary(index).permute(0,3,1,2)
        detached = val.clone().detach()
        detached.requires_grad_(True) # [batch, 128, 9, 9]

        # # reconstruction loss
        f = self.decode(detached)

        loss_rec = F.mse_loss(f,x) * 784 # log-likelihood 

        # nearest neighbor loss
        dict_loss = F.mse_loss(val,z.clone().detach()) * self.latent_channel * self.RATE
        enc_loss = F.mse_loss(z,val.clone().detach()) * self.latent_channel * self.BETA * self.RATE
        # if dict_loss.item() > 20:
        #     print('z.range:',z.min().item(),z.max().item())
        #     print('val.range:',val.min().item(),val.max().item())
        var_loss = torch.zeros((1,)) # - self.dictionary.weight.var(dim=0).mean()

        if do_update:
            self.optimizer.zero_grad()
            loss_rec.backward()
            grad = detached.grad * self.ENCODER_RATE
            # print('\nnorm:',grad.norm(),flush=True)
            # if grad.norm() > .1:
                # grad *= (.1 / grad.norm())
            (z * grad).sum().backward(retain_graph=True) # copy gradient to encoder
            (dict_loss + enc_loss).backward()
            # var_loss.backward()
            # old = self.dictionary.weight.data.clone()
            # print(self.dictionary.weight.grad)
            self.optimizer.step()
            # self.dictionary.weight.data.copy_(self.AVG_GAMMA * old + (1 - self.AVG_GAMMA) * self.dictionary.weight)

        return loss_rec,dict_loss,enc_loss,var_loss,index
    
    @torch.no_grad()
    def generate(self,z:torch.Tensor):
        # print(z.cpu().tolist())
        # print(z.shape)
        # print('dictionary vector variance:',self.dictionary.weight.var(dim=0).mean())
        # print('dictionary:',self.dictionary.weight)
        d = self.decode(self.dictionary(z).permute(0,3,1,2))
        # print(d)
        return d