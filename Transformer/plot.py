import re
import numpy as np
import matplotlib.pyplot as plt
info = {
    'train_losses':[],
    'epochs':[],
    'eval_losses':[],
    'eval_pples':[],
}
keys = {
    r'^\[Train Finished\].*epoch (\d\.\d+).*$':'epochs',
    r'^\[Train Finished\].*loss (\d\.\d+).*$':'train_losses',
    r'^\[Eval Finished\].*loss (\d\.\d+).*$':'eval_losses',
    r'^\[Eval Finished\].*PPL:([^ ]+).*$':'eval_pples',
}
lns = open('our.log').readlines()
for ln in lns:
    for key in keys:
        m = re.findall(key,ln)
        if len(m)>0:
            info[keys[key]].append(float(m[0]))
info['train_losses'] = [9.3825] + info['train_losses'] # the first loss can't be extracted out, I don't know why
info['eval_losses'] = [9.3783] + info['eval_losses']
# print(info)

x = np.array(info['epochs'])-1
plt.plot(
    x,np.array(info['train_losses']),label='train_losses'
)
plt.plot(
    x,np.array(info['eval_losses']),label='eval_losses'
)
# plt.plot(
#     x,np.array(info['eval_pples']),label='eval_pples'
# )
# print([len(x) for y,x in info.items()])
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
plt.savefig('nn.png')