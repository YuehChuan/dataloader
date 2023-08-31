# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Create the dataset with N_SAMPLES samples
N_SAMPLES, D_in, H, D_out = 200, 2, 30, 1

from sklearn.datasets import make_moons, make_blobs
X, y = make_moons(n_samples=N_SAMPLES, noise=0.1)


# visualize in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')
plt.show()

print(X[0][0])
print(X[0][1])

dataList=[]

for xi,yi in(zip(X,y)):
    dataRow = [str(xi[0]), str(xi[1]), str(yi)]
    dataList.append(dataRow)

with open('data.txt', 'a') as f:
    for line in dataList:
        for data in line:
            f.write(data)
            f.write(' ')
        f.write('\n')


# Define the batch size and the number of epochs
BATCH_SIZE = 100
N_EPOCHS = 100
xin = torch.from_numpy(X)
xin=xin.float()
yin=torch.from_numpy(y)
yin=yin.unsqueeze(1)
yin=yin.float()

# Use torch.utils.data to create a DataLoader
# that will take care of creating batches
dataset = TensorDataset(xin, yin)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define model, loss and optimizer
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    #torch.nn.Linear(H, H),
    #torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Sigmoid(),
)

loss_fn = torch.nn.BCELoss(reduction='sum')
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# Get the dataset size for printing (it is equal to N_SAMPLES)
dataset_size = len(dataloader.dataset)

# Loop over epochs
for epoch in range(N_EPOCHS):
    print(f"Epoch {epoch + 1}\n-------------------------------")

    # Loop over batches in an epoch using DataLoader
    for id_batch, (x_batch, y_batch) in enumerate(dataloader):

        y_batch_pred = model(x_batch)

        loss = loss_fn(y_batch_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Every 100 batches, print the loss for this batch
        # as well as the number of examples processed so far
        if id_batch % 100 == 0:
            loss, current = loss.item(), (id_batch + 1)* len(x_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{dataset_size:>5d}]")

torch.save(model, "nn.pth")
