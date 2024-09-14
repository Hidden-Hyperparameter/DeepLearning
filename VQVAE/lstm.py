import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):

    def __init__(self,input_len=4,num_classes=64):
        super().__init__()
        self.embedding_dim = 64
        self.num_classes = num_classes + 1
        self.input_len = input_len
        self.embedding = nn.Embedding(self.num_classes,self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.embedding_dim, batch_first = True)
        self.fc = nn.Linear(self.embedding_dim,self.num_classes)
        self.bos = self.num_classes - 1

    def forward(self,x):
        x = self.embedding(x)
        pred,_ = self.lstm(x)
        logits = self.fc(pred)
        return logits
    
    def get_loss(self,x):
        x_with_bos = torch.cat([torch.ones(x.shape[0],1,dtype=torch.int).to(x.device) * self.bos,x],dim=-1)
        return F.cross_entropy(self(x_with_bos)[:,:-1,:].reshape(-1,self.num_classes),x.reshape(-1))

    def generate(self,num_samples=100):
        x_with_bos = torch.ones(num_samples,1,dtype=torch.int).to(device) * self.bos
        for i in range(self.input_len):
            logits = self(x_with_bos)
            distr = torch.distributions.Categorical(logits=logits[:,-1,:])
            x_with_bos = torch.cat([x_with_bos,distr.sample().unsqueeze(-1)],dim=1)
        return x_with_bos[:,1:]

if __name__ == '__main__':
    model = LSTMModel().to(device)
    x = torch.randint(0,64,(77,4)).to(device)
    u = model.generate()