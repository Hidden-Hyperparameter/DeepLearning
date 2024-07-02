import sys,os
sys.path.append(os.path.abspath('..'))

import torch
import torch.nn.functional as F
# from model import Transformer,device
from nn_model import Transformer,device
from utils import WMT19,count_parameters
from tqdm import tqdm

def generate(model,wmt:WMT19,):
    model.eval()
    valid_loader = wmt.valid_dataloader
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
                    eval_bar.set_description(f'[Valid] Epoch fake: Loss {sum(eval_losses)/len(eval_losses):.4f}, PPL: {ppl:.4f}')
    
    print(f'[Eval Finished] epoch fake, loss {sum(eval_losses)/len(eval_losses):.4f}, PPL:{ppl:.4f} ',flush=True)
    print(wmt.decode(src[2],lang='en'))
    print(wmt.decode(tgt[2],lang='zh'))
    with torch.no_grad():
        src = src[:1]
        generated = model.generate(src) # Chineses
        src = src[0];generated = generated[0]
        print('---Generation---',flush=True)
        print('English:',wmt.decode(src,lang='en'))
        print('(generated) Chinese:',wmt.decode(generated,lang='zh'))
        print('(true) Chinese:',wmt.decode(tgt[0],lang='zh'))

if __name__ == '__main__':
    wmt19 = WMT19(batch_size=12)
    model = torch.load('./models/transformer_014010.pt').to(device)
    count_parameters(model)
    # optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=3e-5)
    generate(model,wmt19)