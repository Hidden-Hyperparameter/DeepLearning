import sys,os
sys.path.append(os.path.abspath('..'))

import torch
import torch.nn.functional as F
from model import Transformer,device
from utils import WMT19,count_parameters
from tqdm import tqdm

def train(epochs,model:Transformer,wmt:WMT19,optimizer):
    train_loader = wmt.train_dataloader
    valid_loader = wmt.valid_dataloader

    for epoch in range(epochs):
        model.train()
        losses = []
        with tqdm(train_loader) as bar:
            for i,(src,tgt) in enumerate(bar):
                src = src.to(device)
                tgt = tgt.to(device)
                tgt_shifted = torch.cat(
                    (tgt[:,1:],wmt.tgt_dict.bos*torch.ones([tgt.shape[0],1]).long().to(device))
                ,dim=-1)
                logits = model(src,tgt)
                loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),tgt_shifted.reshape(-1),ignore_index=wmt.tgt_dict.pad)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if i % 10 == 0:
                    bar.set_description(f'Epoch {epoch+1}: Loss {sum(losses)/len(losses):.4f}')
        
        model.eval()
        losses = []
        pples = []
        with torch.no_grad():
            with tqdm(valid_loader) as bar:
                for src,tgt in bar:
                    src = src.to(device)
                    tgt = tgt.to(device)
                    tgt_shifted = torch.cat(
                        (tgt[:,1:],wmt.tgt_dict.bos*torch.ones([tgt.shape[0],1]).long().to(device))
                    ,dim=-1)
                    logits = model(src,tgt)
                    loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),tgt_shifted.reshape(-1),ignore_index=wmt.tgt_dict.pad)
                    ppl = torch.exp(loss)

                    losses.append(loss.item())
                    pples.append(ppl.item())

                    if i % 10 == 0:
                        bar.set_description(f'[Valid] Epoch {epoch+1}: Loss {sum(losses)/len(losses):.4f}, PPL: {ppl:.4f}')

        # generate
        model.eval()
        with torch.no_grad():
            src = src[:1]
            generated = model.generate(src) # Chineses
            src = src[0];generated = generated[0]
            print('---Generation---')
            print('English:',wmt.decode(src,lang='en'))
            print('(generated) Chinese:',wmt.decode(generated,lang='zh'))
            print('(true) Chinese:',wmt.decode(tgt[0],lang='zh'))

if __name__ == '__main__':
    wmt19 = WMT19()
    model = Transformer(
        src_vocab_size=len(wmt19.src_dict),
        tgt_vocab_size=len(wmt19.tgt_dict),
        src_pad_index=wmt19.src_dict.pad,
        tgt_pad_index=wmt19.tgt_dict.pad,
        bos_index=wmt19.tgt_dict.bos,
        eos_index=wmt19.tgt_dict.eos,
    ).to(device)
    count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    train(20,model,wmt19,optimizer)