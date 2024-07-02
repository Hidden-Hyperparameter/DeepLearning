import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class Transformer(nn.Module):

    def __init__(self,
        src_vocab_size,tgt_vocab_size,
        src_pad_index,tgt_pad_index,bos_index,eos_index,
        embedding_dim = 256,
        hidden_dim = 256,
        num_heads = 8,
        num_layers = 2
    ):
            super().__init__()
            print('Using NN Model...')
            self.nn_transformer = nn.Transformer(
                d_model=embedding_dim,
                nhead=num_heads,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                dim_feedforward=hidden_dim,
                dropout=0,
                batch_first=True
            )
            self.src_pad_index = src_pad_index
            self.tgt_pad_index = tgt_pad_index
            self.src_embedding = nn.Embedding(num_embeddings=src_vocab_size,embedding_dim=embedding_dim)
            self.tgt_embedding = nn.Embedding(num_embeddings=tgt_vocab_size,embedding_dim=embedding_dim)
            self.out_proj = nn.Sequential(
                nn.Linear(embedding_dim,tgt_vocab_size),
            )
            self.bos_index = bos_index
            self.eos_index = eos_index
            self.position_embedding = SinousPositionalEmbedding(size=embedding_dim)

    def forward(self,src,tgt):
        """
        src: input index tensors, corresponing to English source text;
        tgt: (should output) index tensors, corresponing to Chinese target text;
        return: logits of next-token of tgt
        """
        src_embedded = self.src_embedding(src) 
        src_embedded += self.position_embedding(src_embedded)
        tgt_embedded = self.tgt_embedding(tgt) 
        tgt_embedded += self.position_embedding(tgt_embedded)
        tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
        x = self.nn_transformer(
                src=src_embedded,
                tgt=tgt_embedded,
                tgt_is_causal=True,
                tgt_mask=tgt_causal_mask,
                src_key_padding_mask=(src == self.src_pad_index),
                memory_key_padding_mask = (src == self.src_pad_index),
                tgt_key_padding_mask = (tgt == self.tgt_pad_index)
        )
        return self.out_proj(x)
    
    @torch.no_grad()
    def generate(self,src,beam_size=5,max_len=100):
        """
        inputs: src all indices tensor, batched, have padding
        returns: generated indices tensor, batched, have padding
        """
        tgt = torch.tensor([self.bos_index],dtype=torch.long,device=device).reshape(1,1)
        top = [(tgt,0,0,False)]
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
            top = sorted(new_top,key=lambda x:x[1],reverse=True)[:beam_size]
            # print('iteration ',i,top)
        return top[0][0]