import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def read1(dataset,epoch=0): 
    mean = dataset.mean()
    var = dataset.var()
    print('mean',mean,';variance',var)
    random_vec1 = torch.randn_like(dataset[0])
    random_vec2 = torch.randn_like(dataset[0])
    xs = torch.einsum('i,ni->n',random_vec1,dataset)
    ys = torch.einsum('i,ni->n',random_vec2,dataset)
    # make the point smaller
    plt.scatter(xs.detach().cpu().numpy(),ys.detach().cpu().numpy(),s=0.1)
    plt.xlim(-200,200)
    plt.ylim(-200,200)
    # plot a circle
    rad = 28
    theta = np.linspace(0,2*np.pi,rad)
    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    if epoch == 0:
        plt.plot(x,y,label='one std theory')
    plt.legend()
    plt.savefig(os.path.join('./samples',f'plot_epoch_{epoch}.png'))

def read(dataset,epoch=0):
    rad = dataset.norm(dim=-1).detach().cpu().numpy()
    gas = torch.randn_like(dataset)
    gas = gas.norm(dim=-1).detach().cpu().numpy()
    x = np.linspace(0,100,1000)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    for i,val in enumerate(x):
        nextval = x[i+1] if i+1<len(x) else float('inf')
        y[i] += len([1 for r in rad if (val<=r and r <nextval)])
        z[i] += len([1 for r in gas if (val<=r and r <nextval)])
    plt.plot(x,y/len(rad),label='exp')
    plt.plot(x,z/len(gas),label='theory')
    plt.legend()
    
    # plot the theoretical distribution: the radial distribution of isotropic Gaussian
    plt.savefig(os.path.join('./samples',f'rad_epoch_{epoch}.png'))



if __name__  == '__main__':
    dataid = 6
    dataset = torch.load(f'./samples/dataset_{dataid}.pt')
    read(dataset,dataid)