# WaveNet

## Loss doesn't change

Issue: the receptive field is wrong. The casuality is also not correctly implemented.

Reason: the final reshape is wrong.

Issue: the loss decreases to around 1/3 of original value(which corresponds to random guess). But the generation is rubbish (no hearable sounds).

Reason: the mask is wrong! remember we have `A` and `B` type masks, and there should be at least one `A` type mask.

Issue: After this modification, the loss stucks around $2\times 10^5$. The generated audio is not empty, but it is just two "beats". [audio here](./assets/1.wav)

Reason: 
- I think the parameters are not chosen correctly: I should use a larger channel with a smaller kernel size. Otherwise, information may be hard to store in the hidden states. (?)
- I also modified the way of skip connection to better conform the picture in the original paper.

**Notice**: After that, the code has to run with a terminal in which `export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"` is set. However, the model has significantly less parameters than before. I don't know exactly why (from the statics, it seems that this architecture increases the pytorch fragmentation, since the "reserved by PyTorch but unallocated" part is larger). Now the model can only have up to 1M parameters (3M before).

But it still doesn't work...

## So I try to downsample the dataset further (to 4kHz), and then we can increase the model size!

The model can be increased to 4M parameters! The default (random guess) loss becomes $1.1\times 10^5$. 

## The Unconditioned Model

The unconditioned model is trained on (totally) around 150 hours of musics (11 epochs), which cost real-time around 2 hours. The initial loss is $11025\times 5(\text{sec})\times \log 256\approx 3.06\times 10^{5}$, and the valid loss after training for 11 epochs is around $1.00\times 10^5$.

The loss may be further reduced by training for more epochs, but the money and time cost is beyond my budget :(

