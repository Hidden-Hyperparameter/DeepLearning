import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DilatedCausalConv(nn.Conv1d):

    @staticmethod
    def make_causal_mask(size):
        assert size % 2 == 1
        mask = torch.ones(size)
        mask[size//2+1:]=0
        return mask.to(device)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.mask = self.make_causal_mask(kernel_size)

    def forward(self, x):
        with torch.no_grad():
            self.weight.mul_(self.mask)
            # print(self.weight)
        return super().forward(x)

class ResidualBlock(nn.Module):

    def __init__(self,
        channels,
        dilation,
        kernel_size=11,
    ) -> None:
        assert channels % 2 == 0
        super().__init__()
        self.dilatedconv = DilatedCausalConv(
            channels,channels*2,kernel_size,padding=(kernel_size//2) * dilation,
            dilation=dilation
        )
        self.onebyone = nn.Conv1d(channels,channels,1)

    def forward(self, x):
        xc = x.clone()
        x = self.dilatedconv(x)
        x = F.tanh(x[:,::2,...]) * F.sigmoid(x[:,1::2,...])
        x = self.onebyone(x)
        # print(x.shape,xc.shape)
        return x + xc

class WaveNet(nn.Module):

    def __init__(self,
        text_to_audio=False,
        output_dim = 256, # a.k.a. num_classes in regression
        max_in_channel=2, # input can have at most 2 channels
    ):
        super().__init__()
        self.text_to_audio = text_to_audio
        if text_to_audio:
            # language model define here
            raise NotImplementedError()
        # audio model
        channel = 48
        layer_num = 10
        layer_repeates = 3
        self.first = nn.ModuleDict({
            f'first_{i}':nn.Conv1d(i,channel,1) for i in range(1,max_in_channel+1)
        })
        self.layers = []
        for _ in range(layer_repeates):
            self.layers.extend([ResidualBlock(
                channels=channel,
                dilation=2**i
            ) for i in range(layer_num)])
        self.layers = nn.ModuleList(self.layers)
        self.output_dim = output_dim
        self.in_channels = None

        self.mlp = nn.Sequential(
            nn.Conv1d(channel*layer_num*layer_repeates,channel,1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channel,channel,1),
            nn.LeakyReLU(0.2),
        )
        self.project_head = nn.ModuleDict({
            f'project_head_{i}':nn.Conv1d(channel,output_dim*i,1) for i in range(1,max_in_channel+1)
        })

    def quantize(self,audio):
        """
        apply mu-law transformation in paper:
        f(x) = sign(x) * log(1 + mu * |x|) / log(1 + mu)
        where mu=self.output_dim-1, -1 < x < 1. The output is in range (-1,1)
        """
        audio = audio.clamp(-1+1e-5,1-1e-5) # experimentally, this influence is less than 1%
        mu = torch.tensor(self.output_dim - 1).to(device) # = 255
        f = torch.sign(audio) * torch.log(1+mu*torch.abs(audio)) / torch.log(mu+1)
        return (self.output_dim//2 + torch.floor((self.output_dim//2) * f)).to(torch.int)

    def inv_preprocess(self,audio):
        mu = torch.tensor(self.output_dim - 1).to(device)
        f = (audio - (self.output_dim//2)) / (self.output_dim//2)
        x = torch.sign(f) * ((mu+1)**torch.abs(f)-1) / mu
        return x

    def forward(self,audio,tokens=None):
        channel = audio.shape[1]
        size = audio.shape[2]
        if self.text_to_audio:
            raise NotImplementedError()
        audio = self.first[f'first_{channel}'](audio)
        outs = []
        for layer in self.layers:
            audio = layer(audio)
            outs.append(audio.clone())
        outs = torch.cat(outs,dim=1) # skip connection; [batch, num_layers * layer_repeats * channel, audio_leng]
        outs = self.mlp(outs) # [batch, channels, size]
        outs = self.project_head[f'project_head_{channel}'](outs) # [batch, num_classes * in_channels, size]
        return outs.reshape(outs.shape[0],channel,self.output_dim,size).transpose(-1,-2) # [batch, in_channels, size, num_classes]
    
    def get_loss(self,audio,tokens=None):
        outs = self(tokens=tokens,audio=audio)
        # print(torch.softmax(outs,dim=-1)[0][0][:10])
        targets = self.quantize(audio)
        # print(targets[:10])
        return F.cross_entropy(outs.reshape(-1,outs.shape[-1]),targets.reshape(-1).long()) * audio.shape[-1]

    @torch.no_grad()
    def generate(self,audio_leng=None,tokens=None,channel=None):
        if channel is None:
            channel = 1
        if self.text_to_audio:
            if len(tokens.shape) == 1:
                tokens = tokens.unsqueeze(0)
            if audio_leng is None:
                # heuristically set audio_leng
                audio_leng = tokens.shape[1]*...
        
        # generate audio
        audio = torch.zeros(1,channel,audio_leng).to(device)
        with tqdm(range(audio_leng),desc='Generation') as bar:
            for i in bar:
                out = self(audio=audio,tokens=tokens)
                index = torch.argmax(out[0,:,i],dim=-1)
                audio[0,:,i] = self.inv_preprocess(index)   
        return audio[0].detach().cpu()