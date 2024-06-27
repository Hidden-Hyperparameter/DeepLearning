from model import Coupling,device,Flow,Squeeze,float_tp
import torch
from torch.nn.functional import mse_loss
from tqdm import tqdm
cp = Coupling(masktype='A',size=28,channel=1).to(device)
flow = Flow().to(device).to(float_tp)
sq = Squeeze().to(device)

def test_backward():
    x = torch.randn(10,1,28,28).to(device)
    y,logdet = cp(x)
    back = cp.backward(y)
    assert (torch.abs(back-x)<1e-5).all()

def test_logdet():
    x = torch.randn(10,1,28,28).to(device).requires_grad_(True)
    u = torch.zeros([10]).to(device)
    # print('x=',x)
    _,logdet = cp(x)
    # print(logdet,flush=True)
    for i in tqdm(range(28)):
        for j in range(28):
            if x.grad is not None:
                x.grad.zero_()
            cp.zero_grad()
            y,_ = cp(x)
            y[:,0,i,j].sum().backward()
            # print(x.grad[:,0,i,j],flush=True)
            # assert torch.abs(alpha-torch.log(x.grad[:,0,i,j]))<1e-5
            u += torch.log(x.grad[:,0,i,j]).to(device)
    print(logdet-u)
    assert (torch.abs(logdet-u)<1e-4).all()

def test_full_backward():
    x = torch.randn(1,1,28,28).to(device).to(float_tp)
    flow.to(float_tp)
    y,logdet = flow(x)
    back = flow.backward(y)
    assert (torch.abs(back-x)<1e-5).all()

def test_full_logdet():
    batch = 5
    x = torch.randn(batch,1,28,28).to(device).to(float_tp).requires_grad_(True)
    u = torch.zeros([batch,784,784]).to(device).to(float_tp)
    # print(list(flow.modules()))
    torch.autograd.anomaly_mode.set_detect_anomaly(True)
    for param in flow.parameters():
        param.data.normal_(std=0.01)
    
    _,logdet = flow(x)
    print('logdet',logdet)

    for i in tqdm(range(28)):
        # print(f'------{i}------',flush=True)
        for j in range(28):
            if x.grad is not None:
                x.grad.zero_()
            flow.zero_grad()
            y,new_logdet = flow(x)
            print('new_logdet',new_logdet)
            y=y.reshape_as(x)
            y[:,0,i,j].sum().backward()
            # print('x grad',x.grad[:,0,i,j],flush=True)
            # assert torch.abs(alpha-torch.log(x.grad[:,0,i,j]))<1e-5
            # print(x.grad[:,0,...])
            u[:,28*i+j,:] += (x.grad[:,0,...].reshape(u.shape[0],784)).to(device)
            # print('u',u,flush=True)
    print('solving determinant...')
    deter = [torch.linalg.det(u[i]).item() for i in range(u.shape[0])]
    deter = torch.tensor(deter).to(device)
    print(torch.log(deter))
    assert (torch.abs(logdet-torch.log(deter))<1e-4).all()

def d_test():
    x = torch.randn(5,1,28,28).to(device).to(float_tp)
    xp = x + torch.ones_like(x)*0.01
    z,logdet = flow(x)
    zp,logdetp = flow(xp)
    print(torch.norm(z[0]-zp[0])/torch.norm(x[0]-xp[0]))
    print(torch.norm(z[0]))
    print(logdet[0],logdetp[0])

    # singles = [flow(x[i].unsqueeze(0))[-1] for i in range(batch)]
    # assert (torch.abs(torch.cat(singles,dim=-1)-logdet)<1e-5).all()

if __name__ == '__main__':
    # test_backward()
    # print('backward test passed')
    # test_logdet()
    # test_full_backward()
    test_full_logdet()
    # d_test()