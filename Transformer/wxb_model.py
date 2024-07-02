import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import math
import copy
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttention(nn.Module):
    def __init__(self, d, h):
        super(MultiHeadAttention, self).__init__()
        assert d % h == 0, "model dimesion must be divisible by number of heads!!!"
        self.d = d
        self.h = h
        self.d_k = d // h

        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.Wo = nn.Linear(d, d)

    def dot_product_attention(self, Q, K, V, mask = None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim = -1)
        return torch.matmul(attention, V)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.h, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.h * self.d_k)

    def forward(self, Q, K, V, mask = None):
        Q = self.split_heads(self.Wq(Q))
        K = self.split_heads(self.Wk(K))
        V = self.split_heads(self.Wv(V))

        attention_output = self.dot_product_attention(Q, K, V, mask)
        output = self.Wo(self.combine_heads(attention_output))

        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d, hidden_size, activation = nn.ReLU()):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d)
        self.activation = activation

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d).to(device)
        position = torch.arange(0, max_seq_length, dtype = torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d, 2).to(device).float() * -(math.log(10000.0 / d)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0)) # means not going to update during backpropagation

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(nn.Module):
    def __init__(self, d, h, hidden_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d, h)
        self.ff = PositionWiseFeedForward(d, hidden_size)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention))
        output = self.ff(x)
        x = self.norm2(x + self.dropout(output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d, h, hidden_size, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d, h)
        self.cross_attention = MultiHeadAttention(d, h)
        self.ff = PositionWiseFeedForward(d, hidden_size)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attention = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attention))
        attention = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attention))
        output = self.ff(x)
        x = self.norm3(x + self.dropout(output))
        return x

class Transformer(nn.Module):
    def __init__(self, 
        src_vocab_size, tgt_vocab_size,
        src_pad_index,tgt_pad_index,bos_index,eos_index,
         embedding_dim=256,
         hidden_size = 256,
           num_heads=256,
             num_layers=2,  max_seq_length=10000, dropout=0.1):
        super(Transformer, self).__init__()
        print('Using WXB Model')
        self.src_pad_index = src_pad_index
        self.tgt_pad_index = tgt_pad_index
        self.bos_index = bos_index
        self.eos_index = eos_index

        self.encoder_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.positional = PositionalEncoding(embedding_dim, max_seq_length)
        self.encoder = nn.ModuleList(
            [EncoderLayer(embedding_dim, num_heads, hidden_size, dropout) for _ in range(num_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(embedding_dim, num_heads, hidden_size, dropout) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embedding_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != self.src_pad_index).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != self.tgt_pad_index).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        no_peak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length).to(device), diagonal = 1)).bool()
        tgt_mask = tgt_mask & no_peak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedding = self.dropout(self.positional(self.encoder_embedding(src)))
        tgt_embedding = self.dropout(self.positional(self.decoder_embedding(tgt)))
        enc_output = src_embedding
        for enc_layer in self.encoder:
            enc_output = enc_layer(enc_output, src_mask)
        dec_output = tgt_embedding
        for dec_layer in self.decoder:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        return output

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
            top = sorted(new_top,key=lambda x:x[1]/x[2],reverse=True)[:beam_size]
            # print('iteration ',i,top)
        return top[0][0]