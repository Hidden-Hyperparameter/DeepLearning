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
        self.encoder = nn.Sequential(
            nn.Conv2d(1,128,kernel_size=5),
            nn.ReLU(),
            # Residual(128),
            nn.Conv2d(128,128,kernel_size=5,padding=2),
            # nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.ReLU(),
            # Residual(128),
            nn.Conv2d(128,128,kernel_size=5,padding=2),
            nn.MaxPool2d(2),
        )
        # latent space is 128x9x9
        self.feature_conv = nn.Sequential(
            nn.ConvTranspose2d(128,128,kernel_size=4),
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,kernel_size=5),
            nn.ReLU(),
            # nn.ConvTranspose2d(128,128,kernel_size=3),
            # nn.ReLU()
        )
        self.mu_net = nn.Sequential(
            nn.Conv2d(128,1,kernel_size=1),
            # nn.ReLU(),
            # nn.Conv2d(64,1,kernel_size=1),
            # nn.Sigmoid() # image pixel values are between 0 and 1
        )


        self.num_embeddings = 128
        self.dictionary = nn.Embedding(self.num_embeddings,128) # dimension is 128
        self.BETA = .25
        self.RATE = 5
        # self.AVG_GAMMA = 0.99
        self.apply_init()

        self.optimizer = torch.optim.Adam(self.parameters(),lr=3e-4,weight_decay=3e-5)

    def apply_init(self):
        self.dictionary.weight.data.normal_()
        # # apply conv weight init
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def nearest_neighbor(self,z:torch.Tensor):
        # z.shape: [batch,128,9,9]
            
        # z -> [batch, 1, 128, 9, 9]
        # dictionary weight -> [1, self.num_embedding, 128, 1, 1]
        diff = z.unsqueeze(1) - self.dictionary.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [batch, self.num_embedding, 128, 9, 9]
        best = diff.norm(dim=2).argmax(dim=1) # [batch, 9, 9]
        return best

    def update(self,x,do_update=True):
        z = self.encoder(x)
        # print('z.range:',z.min().item(),z.max().item())
        index = self.nearest_neighbor(z) # [batch, 9, 9]
        val = self.dictionary(index).permute(0,3,1,2)
        detached = val.clone().detach().requires_grad_(True) # [batch, 128, 9, 9]

        # reconstruction loss
        f = self.feature_conv(detached)

        loss_rec = F.mse_loss(self.mu_net(f),x)# log-likelihood 

        # nearest neighbor loss
        dict_loss = F.mse_loss(val,z.clone().detach()) * self.RATE
        enc_loss = F.mse_loss(z,val.clone().detach()) * self.BETA * self.RATE
        # if dict_loss.item() > 20:
        #     print('z.range:',z.min().item(),z.max().item())
        #     print('val.range:',val.min().item(),val.max().item())
        var_loss = torch.zeros((1,)) # - self.dictionary.weight.var(dim=0).mean()

        if do_update:
            self.optimizer.zero_grad()
            loss_rec.backward()
            grad = detached.grad
            # print('\nnorm:',grad.norm(),flush=True)
            # if grad.norm() > .1:
            #     grad *= (.1 / grad.norm())
            (z * grad).sum().backward(retain_graph=True) # copy gradient to encoder
            (dict_loss + enc_loss).backward()
            # var_loss.backward()
            # old = self.dictionary.weight.data.clone()
            self.optimizer.step()
            # self.dictionary.weight.data.copy_(self.AVG_GAMMA * old + (1 - self.AVG_GAMMA) * self.dictionary.weight)

        return loss_rec,dict_loss,enc_loss,var_loss,index
    
    @torch.no_grad()
    def generate(self,z:torch.Tensor):
        # print(z.cpu().tolist())
        # print(z.shape)
        print('dictionary vector variance:',self.dictionary.weight.var(dim=0).mean())
        # print('dictionary:',self.dictionary.weight)
        d = self.mu_net(self.feature_conv(self.dictionary(z).permute(0,3,1,2)))
        # print(d)
        return d