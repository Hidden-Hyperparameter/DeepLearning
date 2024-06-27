import torch,torchvision
from torchvision import datasets, transforms
import os

class MNIST:

    def __init__(self,batch_size=128) -> None:
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,))
        ])
        self.batch_size = batch_size

        found = self.auto_find()
        self.dataset = datasets.MNIST(root='./data', train=True, transform=self.transform, download=not found)
        self.test_dataset = datasets.MNIST(root='./data', train=False, transform=self.transform, download=not found)

        train_size = int(0.8 * len(self.dataset))
        valid_size = len(self.dataset) - train_size
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.dataset, [train_size, valid_size])
        
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def auto_find(self):
        if os.path.exists('./data/MNIST'):
            return True
        parent_dir = os.path.dirname(__file__)
        lists = os.listdir(parent_dir)
        for item in lists:
            if os.path.exists(os.path.join(parent_dir,item,'data/MNIST')):
                target_dir = os.path.join(parent_dir,item,'data')
                print(f'[INFO] Using MNIST dataset at {target_dir}')
                os.system(f'cp -r "{target_dir}" .')
                return True
        return False


class CIFAR_10:

    def __init__(self, batch_size=128) -> None:
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.batch_size = batch_size
        self.dataset = datasets.CIFAR10(root='./data', train=True, transform=self.transform, download=True)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        train_size = int(0.8 * len(self.dataset))
        valid_size = len(self.dataset) - train_size
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.dataset, [train_size, valid_size])
        
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataset = datasets.CIFAR10(root='./data', train=False, transform=self.transform, download=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def auto_find(self):
        if os.path.exists('./data/cifar-10-batches-py'):
            return True
        parent_dir = os.path.dirname(__file__)
        lists = os.listdir(parent_dir)
        for item in lists:
            if os.path.exists(os.path.join(parent_dir,item,'data/cifar-10-batches-py')):
                target_dir = os.path.join(parent_dir,item,'data')
                print(f'[INFO] Using MNIST dataset at {target_dir}')
                os.system(f'cp -r "{target_dir}" .')
                return True
        return False
    

def count_parameters(model):
    print('Model Parameters Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))


def save_figure(path,image:torch.Tensor,nrow=16,):
    grid = torchvision.utils.make_grid(image.reshape(-1,1,28,28).cpu(), nrow=nrow)
    torchvision.utils.save_image(grid, path)