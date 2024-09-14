# from pixel import PixelCNN
from lstm import LSTMModel

import torch,tqdm,os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Your device is:',device)

def get_model():
    model = LSTMModel(num_classes=64).to(device)
    # model = PixelCNN(num_class=64).to(device)
    x = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {x:,} trainable parameters')
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    return model,optimizer

def train(data_file,epochs=200):
    model,optimizer = get_model()

    data = torch.load(os.path.join('models',data_file))
    train_loader = torch.utils.data.DataLoader(data,batch_size=512,shuffle=True)
    with tqdm.trange(epochs) as bar:
        for epoch in bar:

            # train
            model.train()
            losses = []
            for x in train_loader:
                x = x.to(device).reshape(x.shape[0],-1)
                loss = model.get_loss(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            bar.set_description(f'Epoch {epoch}/{epochs}, [Train] Loss: {sum(losses[-10:])/len(losses[-10:]):.4f}')

    number = ''.join([c for c in data_file if c.isdigit()])
    torch.save(model.state_dict(),os.path.join('models',f'lstm_ep{number}.pth'))

if __name__ == '__main__':
    train(data_file='data_ep49.pt')