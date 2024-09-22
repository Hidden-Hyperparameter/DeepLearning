import sys,os
sys.path.append(os.path.abspath('..'))

from evaluation import MNIST_uncond_FID

if __name__ == '__main__':
    # model = torch.load('epoch_90.pt').to(device)
    # model.__setattr__('generate',lambda num_samples: sample(model))
    print('FID:',MNIST_uncond_FID(generator=None,folder='./out',repeat=1))
    # print('FID:',MNIST_uncond_FID(generator=None,folder='./samples',repeat=1))