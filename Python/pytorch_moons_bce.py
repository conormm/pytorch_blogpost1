from sklearn.datasets import make_moons
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

torch.__version__

%matplotlib inline

def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    return np.eye(num_classes, dtype='uint8')[y]

X, y_ = make_moons(n_samples=1000, noise=.1)
y = to_categorical(y_, 2)

#plt.scatter(X[:, 0], X[:, 1], c=y_, alpha=.4)

class PrepareData(Dataset):

    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

ds = PrepareData(X=X, y=y)
ds = DataLoader(ds, batch_size=50, shuffle=True)

def torch_acc(preds, true_y):
    "Computes accuracy - presumes inputs are already torch.Tensors/Vars"
    n = preds.size()[0]
    acc = np.round((
            tr.eq(preds.max(1)[1], true_y).sum().data[0] / n), 3)
    return acc

class MoonsModel(nn.Module):
    def __init__(self, n_features, n_neurons):
        super(MoonsModel, self).__init__()
        self.hidden = nn.Linear(in_features=n_features, out_features=n_neurons)
        self.out_layer = nn.Linear(in_features=n_neurons, out_features=2)

    def forward(self, X):
        out = F.relu(self.hidden(X))
        out = F.sigmoid(self.out_layer(out))
        return out

model = MoonsModel(n_features=2, n_neurons=50)

cost_func = nn.BCELoss()
optimizer = tr.optim.Adam(params=model.parameters(), lr=0.01)

num_epochs = 20

losses = []
accs = []
for e in range(num_epochs):

    for ix, (_x, _y) in enumerate(ds):

        #=========make inpur differentiable=======================

        _x = Variable(_x).float()
        _y = Variable(_y).float()

        #========forward pass=====================================
        yhat = model(_x).float()
        loss = cost_func(yhat, _y)
        acc = tr.eq(yhat.round(), _y).float().mean() # accuracy

        #=======backward pass=====================================
        optimizer.zero_grad() # zero the gradients on each pass before the update
        loss.backward() # backpropagate the loss through the model
        optimizer.step() # update the gradients w.r.t the loss

        losses.append(loss.data[0])
        accs.append(acc.data[0])

    if e % 1 == 0:
        print("[{}/{}], loss: {} acc: {}".format(e,
        num_epochs, np.round(loss.data[0], 3), np.round(acc.data[0], 3)))


pd.DataFrame(dict(losses=losses, acc=accs)).to_csv("Data/model_performance.csv")
