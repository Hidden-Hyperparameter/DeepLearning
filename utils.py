import torch
from torchvision import datasets, transforms

class MNIST:
    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.dataset = datasets.MNIST(root='./data', train=True, transform=self.transform, download=True)

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=64, shuffle=True)

        train_size = int(0.8 * len(self.dataset))
        valid_size = len(self.dataset) - train_size
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.dataset, [train_size, valid_size])
        
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=64, shuffle=False)

        self.test_dataset = datasets.MNIST(root='./data', train=False, transform=self.transform, download=True)

        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=64, shuffle=False)
    