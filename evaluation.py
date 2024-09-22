# This file is partially adapted from Coding Project 3 of the Deep Learning Course at IIIS, Tsinghua University.

from scipy.linalg import sqrtm
import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np 
import os

class MnistInceptionV3(nn.Module):

    def __init__(self, in_channels=3):
        super(MnistInceptionV3, self).__init__()

        self.model = models.inception_v3(pretrained=True)

        # Change the output layer to output 10 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.model(x)
    
def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}, you should choose another numpy or scipy version. Good luck on that!'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) -
            2 * tr_covmean)

image_folder_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.1307, ),
                         (0.3081, )),
])

@torch.no_grad()
def get_generation(generator:nn.Module, folder):
    if isinstance(folder,str) and os.path.exists(folder):
        # use the folder as generated images
        try:
            return ImageFolder(folder,transform=image_folder_transforms)
        except:
            raise ValueError("""Found an invalid folder structure. Make sure the folder is like this:
root/
    class1/
        img1.png
        img2.png
        ...
    class2/
        img1.png
        img2.png
        ...
    ...
                             
Where `root` is the folder you passed in.""")
    else:
        assert hasattr(generator,'generate'), 'Generator should have the `generate` method, which can take a `num_samples` argument and return a batch of generated images.'
        out = []
        for _ in tqdm.trange(10,desc='Generating images'):
            samples = generator.generate(100)
            out.append(samples.cpu())
        return torch.cat(out,dim=0)

def MNIST_uncond_FID(generator:nn.Module=None,folder=None,repeat:int=1):
    """
    Calculate unconditional FID score for MNIST dataset.

    Args:
        generator: Generator model that can generate images.
        folder: Folder that contains generated images. (optional if generator is provided)
        repeat: Number of times to calculate FID score.

    Returns:
        float: FID score.
    """
    from utils import MNIST
    mnist = MNIST(custom_transform=image_folder_transforms)
    model = MnistInceptionV3()
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),"MnistInceptionV3.pth")))

    model.model.fc = nn.Identity()
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
        'cpu')
    model = model.to(device)

    fids = np.zeros((repeat,))

    dataset = get_generation(generator,folder)
    for repeat in tqdm.trange(repeat):
        with torch.no_grad():
            # from image folder
            generated_dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=100,
                shuffle=True,
                pin_memory=True)
            generated_img = None
            for img,_ in generated_dataloader:
                generated_img = img.to(device)
                break

            mnist_dataloader = mnist.test_dataloader
            mnist_img = None
            for img, _ in mnist_dataloader:
                mnist_img = img.to(device)
                break

            # calculate activations
            act1 = model(mnist_img).cpu().numpy()
            act2 = model(generated_img).cpu().numpy()
            # calculate mean and covariance statistics
            mu1, sigma1 = act1.mean(0), np.cov(act1, rowvar=False)
            mu2, sigma2 = act2.mean(0), np.cov(act2, rowvar=False)
            fid = _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

            fids[repeat] = fid
            torch.cuda.empty_cache()
    return fids.mean()