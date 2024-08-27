import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,128,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3),
            nn.ReLU(),
            # nn.Conv2d(128,128,kernel_size=3,padding=1),
            # nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3),
        )
        # latent space is 128x9x9
        self.feature_conv = nn.Sequential(
            nn.ConvTranspose2d(128,128,kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,kernel_size=3,stride=2),
            nn.ReLU(),
            # nn.ConvTranspose2d(128,128,kernel_size=3,stride=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(128,128,kernel_size=4),
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,kernel_size=3),
            nn.ReLU()
        )
        self.mu_net = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64,1,kernel_size=1),
            # nn.Sigmoid() # image pixel values are between 0 and 1
        )

        self.num_embeddings = 8
        self.dictionary = nn.Embedding(self.num_embeddings,128) # 128 dictionary vectors
        self.BETA = 1
        self.RATE = 1e-2
        self.apply_init()

        self.optimizer = torch.optim.Adam(self.parameters(),lr=3e-4)

    def apply_init(self):
        self.dictionary.weight.data.normal_()
        # # apply conv weight init
        # for m in self.modules():
        #     if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        #         nn.init.kaiming_normal_(m.weight)
        #         nn.init.zeros_(m.bias)

    @torch.no_grad()
    def nearest_neighbor(self,z:torch.Tensor):
        # z.shape: [batch,128,9,9]
            
        # z -> [batch, 1, 128, 9, 9]
        # dictionary weight -> [1, self.num_embedding, 128, 1, 1]
        diff = z.unsqueeze(1) - self.dictionary.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [batch, self.num_embedding, 128, 9, 9]
        best = diff.norm(dim=2).argmax(dim=1) # [batch, 9, 9]
        return best

    def update(self,x,do_update=True):
        if do_update:self.optimizer.zero_grad()

        z = self.encoder(x)
        index= self.nearest_neighbor(z)
        val = self.dictionary(index).permute(0,3,1,2)
        detached = val.clone().detach().requires_grad_(True) # [batch, 128, 4, 4]

        # reconstruction loss
        f = self.feature_conv(detached)

        # distr = torch.distributions.Normal(self.mu_net(f),torch.exp(self.sigma_net(f)))
        # loss_rec =  - distr.log_prob(x).mean()
        loss_rec = F.mse_loss(self.mu_net(f),x)
        if do_update:
            loss_rec.backward()
            (z * detached.grad).sum().backward(retain_graph=True) # propogate gradient to encoder

        # nearest neighbor loss
        dict_loss = F.mse_loss(val,z.clone().detach()) * self.RATE
        enc_loss = F.mse_loss(z,val.clone().detach()) * self.BETA * self.RATE
        var_loss = torch.zeros((1,)) # - self.dictionary.weight.var(dim=0).mean()
        if do_update: 
            (dict_loss + enc_loss).backward()
            # var_loss.backward()
            self.optimizer.step()

        return loss_rec,dict_loss,enc_loss,var_loss,index
    
    @torch.no_grad()
    def generate(self,z:torch.Tensor):
        # print(z.cpu().tolist())
        # print(z.shape)
        print('dictionary vector variance:',self.dictionary.weight.var(dim=0).mean())
        d = self.mu_net(self.feature_conv(self.dictionary(z).permute(0,3,1,2)))
        # print(d)
        return d