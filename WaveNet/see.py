import torchaudio,torch
from audio_model import WaveNet,device
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.abspath('..'))
from utils import saveaudio
# x,y = torchaudio.load('./samples/corrupted.wav')
# z,w = torchaudio.load('./samples/recovered.wav')
# print(((x-z)/x)[...,11025:].tolist())

from audio_model import WaveNet,device
from train import visualize_receptive_field,sample
model = WaveNet().to(device)
# model = torch.load('./models/model_74.1_hours.pt').to(device)
# visualize_receptive_field(model)

# sample a audio and save
sample(model,save_dir='./samples/init.wav',init=True)