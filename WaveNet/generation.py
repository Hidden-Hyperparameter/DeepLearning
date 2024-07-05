import torch
from train import sample,device

if __name__ == '__main__':
    model = torch.load('./models/model_74.2_hours.pt').to(device)
    sample(model,save_dir='./samples/74.2_hours.wav')