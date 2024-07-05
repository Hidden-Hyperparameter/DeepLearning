import torchaudio,torch
from audio_model import WaveNet,device
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.abspath('..'))
from utils import saveaudio
# net = WaveNet().to(device)
net:WaveNet = torch.load('./models/model_74.1_hours.pt').to(device)
x,y = torchaudio.load('./samples/original_Electronic.wav')
x = x[...,:11025+1000] # get first 2 seconds
saveaudio(x,sample_rate=11025,path='./samples/ground_truth.wav')
x = x.to(device)
valid_len  = 11025
x_corrputed = x.clone()
x_corrputed[...,valid_len:].normal_(0,0.4)
x_corrputed.clamp_(-1,1)
assert (x_corrputed-x)[...,:valid_len].abs().max() < 1e-5
saveaudio(x_corrputed,sample_rate=11025,path='./samples/corrupted.wav')
print('Impainting....')
# x,y = torchaudio.load('./samples/148.2_hours.wav')
# print('loss:',net.get_loss(audio=x.unsqueeze(0))/x.shape[-1])
x_recovered = net.impaint(audio=x_corrputed,valid_len=valid_len,debug_info={
    # 'x':x,
})
saveaudio(x_recovered,sample_rate=11025,path='./samples/recovered.wav')
# # print(x)
# # print(x.min(),x.max())
# print(x)
# z = net.quantize(x)
# l = z.cpu().tolist()
# print(l)
# plt.hist(l,bins=256)
# plt.show()
# plt.savefig('./quantized.png')
# # xp = net.inv_preprocess(z)
# print(xp)
# assert (x-xp).abs().max() < 1e-5