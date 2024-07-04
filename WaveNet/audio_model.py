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
        dict_size=0,
        in_channels=1,
        embedding_dim=256,
        output_dim = 256, # a.k.a. num_classes in regression
    ):
        super().__init__()
        self.text_to_audio = text_to_audio
        if text_to_audio:
            # language model
            self.lang_embedding = nn.Embedding(dict_size,embedding_dim)
            self.lang_model = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=embedding_dim,
                num_layers=3,
                batch_first=True,
            )
        # audio model
        channel = 32
        layer_num = 10
        layer_repeates = 3
        self.first = nn.Conv1d(in_channels,channel,1)
        self.layers = []
        for _ in range(layer_repeates):
            self.layers.extend([ResidualBlock(
                channels=channel,
                dilation=2**i
            ) for i in range(layer_num)])
        self.layers = nn.ModuleList(self.layers)
        self.output_dim = output_dim
        self.in_channels = in_channels

        self.mlp = nn.Sequential(
            nn.Conv1d(channel*layer_num*layer_repeates,channel,1),
            nn.ReLU(),
            nn.Conv1d(channel,channel,1),
            nn.ReLU(),
            nn.Conv1d(channel,output_dim * in_channels,1)
        )

    def quantize(self,audio):
        """
        apply mu-law transformation in paper:
        f(x) = sign(x) * log(1 + mu * |x|) / log(1 + mu)
        where mu=self.output_dim-1, -1 < x < 1. The output is in range (-1,1)
        """
        mu = self.output_dim - 1
        f = torch.sign(audio) * torch.log(1+mu*torch.abs(audio)) / torch.log(mu+1)
        return (self.output_dim//2 + torch.floor((self.output_dim//2) * f)).to(torch.int)

    def inv_preprocess(self,audio):
        mu = self.output_dim - 1
        f = (audio - (self.output_dim//2)) / (self.output_dim//2)
        x = torch.sign(f) * ((mu+1)**torch.abs(f)-1) / mu
        return x

    def forward(self,audio,tokens=None):
        if self.text_to_audio:
            tokens = self.lang_embedding(tokens)
            lang_features,(h,c) = self.lang_model(tokens) # lang_features shape: (batch,seq, 3*embedding_dim)
        audio = self.first(audio)
        outs = []
        for layer in self.layers:
            audio = layer(audio)
            outs.append(audio.clone())
        outs = torch.cat(outs,dim=1) # skip connection; [batch, num_layers * layer_repeats * channel, audio_leng]
        outs = self.mlp(outs) # [batch, num_classes * in_channels, size]
        return outs.reshape(outs.shape[0],self.in_channels,-1,self.output_dim) # [batch, in_channels, size, num_classes]
    
    def get_loss(self,audio,tokens=None):
        outs = self(tokens,audio)
        targets = self.quantize(audio)
        return F.cross_entropy(outs.reshape(-1,outs.shape[-1]),targets.reshape(-1))

    @torch.no_grad()
    def generate(self,audio_leng=None,tokens=None):
        if self.text_to_audio:
            if len(tokens.shape) == 1:
                tokens = tokens.unsqueeze(0)
            if audio_leng is None:
                # heuristically set audio_leng
                audio_leng = tokens.shape[1]*...
        
        # generate audio
        audio = torch.zeros(1,self.in_channels,audio_leng).to(device)
        with tqdm(range(audio_leng),desc='Generation') as bar:
            for i in bar:
                out = self(audio=audio,tokens=tokens)
                index = torch.argmax(out[0,:,i],dim=-1)
                audio[0,:,i] = self.inv_preprocess(index)   
        return audio.unsqueeze(0)