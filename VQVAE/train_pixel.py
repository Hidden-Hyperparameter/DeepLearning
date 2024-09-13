from pixel import PixelCNN

import torch,tqdm,os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Your device is:',device)

def get_model():
    model = PixelCNN(num_class=64).to(device)
    x = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {x:,} trainable parameters')
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=3e-5)
    return model,optimizer

def train(data_file,epochs=70):
    model,optimizer = get_model()

    data = torch.load(os.path.join('models',data_file))
    train_loader = torch.utils.data.DataLoader(data,batch_size=256,shuffle=True)
    test_loader = torch.utils.data.DataLoader(data,batch_size=256,shuffle=False)
    for epoch in range(epochs):

        # train
        model.train()
        bar = tqdm.tqdm(train_loader)
        losses = []
        for x in bar:
            x = x.to(device)
            loss = model.get_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            bar.set_description(f'Epoch {epoch}/{epochs}, Loss: {sum(losses[-10:])/len(losses[-10:]):.4f}')
        bar.close()

        # eval
        model.eval()
        with torch.no_grad():
            losses =  []
            for x in test_loader:
                x = x.to(device)
                loss = model.get_loss(x)
                losses.append(loss.item())
            mean = sum(losses)/len(losses)
            print(f'[Valid] Epoch {epoch}/{epochs}, Loss: {mean:.4f}')

    number = ''.join([c for c in data_file if c.isdigit()])
    torch.save(model.state_dict(),os.path.join('models',f'pixelcnn_ep{number}.pth'))

if __name__ == '__main__':
    train(data_file='data_ep33.pt')