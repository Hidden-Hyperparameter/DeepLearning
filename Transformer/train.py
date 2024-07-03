import sys,os
sys.path.append(os.path.abspath('..'))

import torch
import torch.nn.functional as F
# from model import Transformer,device
from nn_model import Transformer,device
from utils import WMT19,count_parameters
from tqdm import tqdm

def train(epochs,model:Transformer,wmt:WMT19,optimizer,eval_num=500):
    train_loader = wmt.train_dataloader
    valid_loader = wmt.valid_dataloader
    leng = len(train_loader)
    tot_trained = 0

    for epoch in range(epochs):
        model.train()
        losses = []
        bar = tqdm(total=eval_num)
        for i,(src,tgt) in enumerate(train_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_shifted = torch.cat(
                (tgt[:,1:],wmt.tgt_dict.eos*torch.ones([tgt.shape[0],1]).long().to(device))
            ,dim=-1)
            logits = model(src,tgt)
            loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),tgt_shifted.reshape(-1),ignore_index=wmt.tgt_dict.pad)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if i % 10 == 0:
                tot_trained += 10
                bar.update(10)
                bar.set_description(f'Epoch {i/leng+epoch+1:.4f}(tot:{tot_trained}): Loss {sum(losses[-10:])/len(losses[-10:]):.4f}')
    
            if (i) % eval_num == 0:
                # set progress bar to 0
                bar.close()
                print(f'[Train Finished] epoch {i/leng+epoch+1:.4f}(tot {tot_trained}), loss {sum(losses)/len(losses):.4f} ',flush=True)
                bar = tqdm(total=eval_num)
                # reset old
                losses = []
                pples = []

                # evaluation
                model.eval()
                eval_losses = []
                eval_pples = []
                with torch.no_grad():
                    with tqdm(valid_loader) as eval_bar:
                        for eval_i,(src,tgt) in enumerate(eval_bar):
                            src = src.to(device)
                            tgt = tgt.to(device)
                            tgt_shifted = torch.cat(
                                (tgt[:,1:],wmt.tgt_dict.eos*torch.ones([tgt.shape[0],1]).long().to(device))
                            ,dim=-1)
                            logits = model(src,tgt)
                            loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),tgt_shifted.reshape(-1),ignore_index=wmt.tgt_dict.pad)
                            ppl = torch.exp(loss)

                            eval_losses.append(loss.item())
                            eval_pples.append(ppl.item())

                            if eval_i % 10 == 0:
                                eval_bar.set_description(f'[Valid] Epoch {epoch+1}: Loss {sum(eval_losses)/len(eval_losses):.4f}, PPL: {ppl:.4f}')
                
                print(f'[Eval Finished] epoch {i/leng+epoch+1:.4f}, loss {sum(eval_losses)/len(eval_losses):.4f}, PPL:{ppl:.4f} ',flush=True)

                # generate
                model.eval()
                with torch.no_grad():
                    src = src[:1]
                    generated = model.generate(src) # Chineses
                    src = src[0];generated = generated[0]
                    print('---Generation---',flush=True)
                    print('English:',wmt.decode(src,lang='en'))
                    print('(generated) Chinese:',wmt.decode(generated,lang='zh'))
                    print('(true) Chinese:',wmt.decode(tgt[0],lang='zh'))

                # save model
                torch.save(model,f'./models/transformer_{tot_trained:06d}.pt')

if __name__ == '__main__':
    wmt19 = WMT19(batch_size=64)
    model = Transformer(
        src_vocab_size=len(wmt19.src_dict),
        tgt_vocab_size=len(wmt19.tgt_dict),
        src_pad_index=wmt19.src_dict.pad,
        tgt_pad_index=wmt19.tgt_dict.pad,
        bos_index=wmt19.tgt_dict.bos,
        eos_index=wmt19.tgt_dict.eos,
    ).to(device)
    count_parameters(model)
    # optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    train(20,model,wmt19,optimizer)