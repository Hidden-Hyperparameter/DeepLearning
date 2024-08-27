import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

try:
    from image_datasets import MNIST,CIFAR_10,save_figure,train_generative_model
except: pass
try:
    from text_datasets import WMT19
except: pass
try:
    from audio_datasets import AISHELL_3,saveaudio,MusicGenres
except: pass

def count_parameters(model):
    print('Model Parameters Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))