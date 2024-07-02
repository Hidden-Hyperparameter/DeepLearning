import torch
import torch.nn as nn
import torch.nn.functional as F
# device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention(nn.Module):

    def __init__(self,embed_dim,hidden_dim,out_dim,num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.head_out_dim = out_dim // num_heads
        self.q_net = nn.Linear(embed_dim,hidden_dim)
        self.k_net = nn.Linear(embed_dim,hidden_dim)
        self.v_net = nn.Linear(embed_dim,out_dim)

    def forward(self,problem,context,mask=None):
        """
        problem: to calculate q. shape: (batch, p_leng, embed_dim)
        context: to calculate k,v. shape: (batch, c_leng, embed_dim)
        mask: attn mask, shape: (batch, p_leng, c_leng)
            - explain: mask[i,j]=-inf then problem i doesn't attend to context j

        output.shape: (batch, p_leng, out_dim)
        """
        batch = problem.shape[0]
        q_heads = self.q_net(problem).reshape([batch,-1,self.head_dim,self.num_heads])
        k_heads = self.k_net(context).reshape([batch,-1,self.head_dim,self.num_heads])
        v_heads = self.v_net(context).reshape([batch,-1,self.head_out_dim,self.num_heads])
        q_dot_k = torch.einsum('bidh,bjdh->bijh',q_heads,k_heads)/(self.head_dim**0.5)
        q_dot_k = torch.softmax(q_dot_k,dim=2)
        if mask is not None:
            q_dot_k = q_dot_k + mask.unsqueeze(-1)
        attn_out = torch.einsum('bijh,bjoh->bioh',q_dot_k,v_heads)
        return attn_out.reshape(batch,-1,self.head_out_dim*self.num_heads)

class SinousPositionalEmbedding(nn.Module):

    def __init__(self,size):
        super().__init__()
        self.register_buffer(
            'fractions',10000**(torch.arange(0,size,2)/size).to(device),
            persistent=False
        )

    @torch.no_grad()
    def forward(self,x):
        batch,leng,embed_dim = x.shape
        angles = self.fractions.reshape(1,-1) * torch.arange(leng).to(device).reshape(-1,1) # (leng, size/2)
        out = torch.zeros(leng,embed_dim).to(device)
        out[:,::2] = torch.sin(angles)
        out[:,1::2] = torch.cos(angles)
        return out.expand(batch,*out.shape)


class EncoderLayer(nn.Module):

    def __init__(self,embed_dim,hidden_dim,num_heads):
        super().__init__()
        self.attn = Attention(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            out_dim=embed_dim
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self,x,mask=None):
        xc = x.clone()
        x = self.attn(x,x,mask)
        x = self.ln1(x) + xc

        xc = x.clone()
        x = self.mlp(x)
        return self.ln2(x) + xc


class DecoderLayer(nn.Module):

    def __init__(self,embed_dim,hidden_dim,num_heads):
        super().__init__()
        self.self_attn = Attention(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            out_dim=embed_dim
        )
        self.cross_attn = Attention(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            out_dim=embed_dim
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)

    def forward(self,x,context,causal_mask=None,context_pad_mask=None,problem_pad_mask=None):
        if causal_mask is None and problem_pad_mask is None:
            self_attn_mask = None
        elif causal_mask is not None and problem_pad_mask is not None:
            self_attn_mask = causal_mask + problem_pad_mask
        else:
            self_attn_mask = causal_mask or problem_pad_mask

        xc = x.clone()
        x = self.self_attn(x,x,self_attn_mask)
        x = self.ln1(x) + xc

        xc = x.clone()
        x = self.cross_attn(x,context,context_pad_mask)
        x = self.ln2(x) + xc

        xc = x.clone()
        x = self.mlp(x)
        return self.ln3(x) + xc
    
class Transformer(nn.Module):

    def __init__(self,
        src_vocab_size,tgt_vocab_size,
        src_pad_index,tgt_pad_index,bos_index,eos_index,
        embedding_dim = 256,
        hidden_dim = 256,
        num_heads = 8,
        num_layers = 2,
    ):
        super().__init__()
        self.src_embedding = nn.Embedding(num_embeddings=src_vocab_size,embedding_dim=embedding_dim)
        self.tgt_embedding = nn.Embedding(num_embeddings=tgt_vocab_size,embedding_dim=embedding_dim)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(
                embed_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            ) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(
                embed_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            ) for _ in range(num_layers)]
        )
        self.position_embedding = SinousPositionalEmbedding(size=embedding_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(embedding_dim,tgt_vocab_size),
        )
        self.src_pad_index = src_pad_index
        self.tgt_pad_index = tgt_pad_index
        self.bos_index = bos_index
        self.eos_index = eos_index

    @staticmethod
    def make_encoder_mask(src,pad_index):
        """mask out all padding tokens in src"""
        batch,leng = src.shape[:2]
        mask = torch.zeros([batch,leng,leng]).to(device)
        is_pad = (src==pad_index).float().reshape(batch,1,leng).expand(batch,leng,leng)
        mask += is_pad * (-1e6)
        return mask # [batch,leng,leng]

    @staticmethod
    def make_decoder_mask(src,src_pad_index,tgt,tgt_pad_index):
        """mask out all padding tokens in tgt, and make causal mask"""
        tgt_leng = tgt.shape[1]
        enc_mask = Transformer.make_encoder_mask(src,src_pad_index)
        src_mask = enc_mask[:,:1,:].expand(tgt.shape[0],tgt_leng,src.shape[-1])

        pad_mask = Transformer.make_encoder_mask(tgt,tgt_pad_index)
        causal_mask = torch.tril(torch.ones([tgt.shape[1],tgt.shape[1]]).to(device)).float() # i>j is 1, other 0
        causal_mask = (1-causal_mask)*(-1e6)
        return pad_mask,causal_mask,src_mask

    def forward(self,src,tgt):
        """inputs: src all indices tensor, batched, have padding"""
        enc_mask = self.make_encoder_mask(src,self.src_pad_index)
        src_embedded = self.src_embedding(src)
        x = src_embedded + self.position_embedding(src_embedded)
        x_list = []
        for layer in self.encoder_layers:
            x = layer(x,mask=enc_mask)
            x_list.append(x.clone())
        
        pad_mask,causal_mask,context_mask = self.make_decoder_mask(src,self.src_pad_index,tgt,self.tgt_pad_index)
        tgt_embedded = self.tgt_embedding(tgt)
        x = tgt_embedded + self.position_embedding(tgt_embedded)
        for i,layer in enumerate(self.decoder_layers):
            x = layer(x,x_list[i],causal_mask=causal_mask,context_pad_mask=context_mask,problem_pad_mask=pad_mask)
        out = self.out_proj(x)

        return out

    @torch.no_grad()
    def generate(self,src,beam_size=5,max_len=100):
        """
        inputs: src all indices tensor, batched, have padding
        returns: generated indices tensor, batched, have padding
        """
        tgt = torch.tensor([self.bos_index],dtype=torch.long,device=device).reshape(1,1)
        top = [(tgt,0,1,False)]
        for i in range(max_len):
            new_top = []
            for item,score,l,done in top:
                if done:
                    new_top.append((item,score,l,done))
                    continue
                logits = self(src,item) # shape: [1,tgt_leng,tgt_vocab_size]
                log_probs = torch.log(torch.softmax(logits[0,-1],dim=0))
                topk = torch.topk(log_probs,beam_size)
                values = topk.values.detach().cpu().tolist()
                indices = topk.indices.detach().cpu().tolist()
                for j in range(beam_size):
                    new_item = torch.cat(
                        (item,torch.tensor([indices[j]],dtype=torch.long,device=device).reshape(1,1))
                    ,dim=-1)
                    new_score = score + values[j]
                    new_top.append((new_item,new_score,l+1,indices[j]==self.eos_index))
            # print(len(new_top))
            top = sorted(new_top,key=lambda x:x[1]/x[2],reverse=True)[:beam_size]
            # print('iteration ',i,top)
        return top[0][0]