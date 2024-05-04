import re
"""
epoch 0, loss 0.5829: 100%|█████████████████████████████████████████████████████████████████████| 375/375 [00:33<00:00, 11.25it/s]
epoch 0, valid loss 0.4068: 100%|█████████████████████████████████████████████████████████████████| 94/94 [00:03<00:00, 24.36it/s]
"""
s = open('./training.log').readlines()
results = []
for l in s:
    res1 = re.findall(r', loss ([\d]+\.[\d]+)\:',l)
    res2 = re.findall(r', valid loss ([\d]+\.[\d]+)\:',l)
    if len(res1):
        results.append(('train',float(res1[0])))
    if len(res2):
        results.append(('valid',float(res2[0])))
pre = 'valid'
traines = []
valides = []
for x,y in results:
    print(x,y)
    if x == pre:
        continue
    else:
        pre = x
        if x == 'train':
            traines.append(y)
        else:
            valides.append(y)
traines = traines[:-1]
print(len(traines))
print(len(valides))
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,len(traines),1)
plt.plot(x,np.array(traines),label='train loss')
plt.plot(x,np.array(valides),label='valid loss')
plt.legend()
# plt.show()
plt.savefig('./train_curve.png')