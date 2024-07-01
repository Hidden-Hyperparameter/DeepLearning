import torch,torchvision
from torchvision import datasets, transforms
import os
from tqdm import tqdm
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

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
    os.makedirs(os.path.dirname(path),exist_ok=True)
    grid = torchvision.utils.make_grid(image.reshape(-1,1,28,28).cpu(), nrow=nrow)
    torchvision.utils.save_image(grid, path)

def train_generative_model(epochs,model,optimizer,sample_func,conditional=False,eval_interval=1,save_model=False,train_loader=None,valid_loader=None,save_dir='./samples'):
    try:
        device = model.device
    except AttributeError:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if train_loader is None:
        mnist = MNIST(batch_size=256)
        train_loader = mnist.train_dataloader
        valid_loader = mnist.valid_dataloader
    assert valid_loader is not None

    for epoch in range(epochs):
        losses = []
        model.train()
        with tqdm(train_loader) as bar:
            for x,y in bar:
                x = x.to(device); y = y.to(device)
                if conditional:
                    loss = model.get_loss(x,y)
                else:
                    loss = model.get_loss(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                bar.set_description(f'Epoch: {epoch} Loss: {sum(losses)/len(losses)}')
       
        losses = []
        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader) as bar:
                for x,y in bar:
                    x = x.to(device); y = y.to(device)
                    if conditional:
                        loss = model.get_loss(x,y)
                    else:
                        loss = model.get_loss(x)
                    losses.append(loss.item())
                    bar.set_description(f'Epoch: {epoch} [Valid]Loss: {sum(losses)/len(losses)}')

        if (epoch+1) % eval_interval == 0:
            sample_func(model,save_dir=os.path.join(save_dir,f'epoch_{epoch}.png'))
            if save_model:
                torch.save(model,os.path.join(save_dir,f'epoch_{epoch}.pt'))