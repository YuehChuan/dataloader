# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt


N_SAMPLES, D_in, H, D_out = 2000, 2, 30, 1

from sklearn.datasets import make_moons, make_blobs
X, y = make_moons(n_samples=N_SAMPLES, noise=0.1)


plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')
plt.show()


# Define the batch size and the number of epochs
xin = torch.from_numpy(X)
xin=xin.float()
yin=torch.from_numpy(y)
yin=yin.unsqueeze(1)
yin=yin.float()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load("C:/Users/user/source/repos/code/gpt/nn.pth")
model.eval()

ypred=model(xin)

correct=0
for yp, y in list(zip(ypred, yin)):
    if yp>=0.5 and y==1:
        correct=correct+1
    elif yp<0.5 and y==0:
        correct=correct+1
    else:
        pass
print("correct "+str(correct))
print("accuracy: "+str(correct/len(yin)))







