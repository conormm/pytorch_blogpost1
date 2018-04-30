from sklearn.datasets import make_moons
import torch as tr
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from torch.utils.data import dataset, dataloader

%matplotlib inline

X, y_ = make_moons(n_samples=1000, noise=.1)
y = np_utils.to_categorical(y_)

#plt.scatter(X[:, 0], X[:, 1], c=y_, alpha=.4)

def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    return np.eye(num_classes, dtype='uint8')[y]

class PrepareData(dataset):

    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = Variable(torch.from_numpy(X))
        if not torch.is_tensor(y):
            self.y = Variable(torch.from_numpy(y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def torch_acc(preds, true_y):
    "Computes accuracy - presumes inputs are already torch.Tensors/Vars"
    n = preds.size()[0]
    acc = np.round((
            tr.eq(preds.max(1)[1], true_y).sum().data[0] / n), 3)
    return acc

class MoonsModel(nn.Module):
    def __init__(self):
        super(MoonsModel, self).__init__()
        self.hidden_1 = nn.Linear(in_features=2, out_features=50)
        self.relu_h1 = nn.Tanh()
        self.dropout_05 = nn.Dropout(p=0.5)
        self.out_layer = nn.Linear(in_features=50, out_features=2)
        self.sig = nn.Sigmoid()

    def forward(self, X):
        out = self.hidden_1(X)
        out = self.relu_h1(out)
        out = self.dropout_05(out)
        out = self.out_layer(out)
        out = self.sig(out)
        return out

m = MoonsModel()

cost_func = nn.BCELoss()
optimizer = tr.optim.Adam(params=m.parameters(), lr=0.05)

num_epochs = 500

for e in range(num_epochs):

    #========torchify inputs/target============================
    X_ = Variable(tr.from_numpy(X), requires_grad=False).float()
    y_ = Variable(tr.from_numpy(y), requires_grad=False).float()

    #========forward pass=====================================
    yhat = m(X_)
    loss = cost_func(yhat, y_) # loss is probabilty that predicted==1
    acc = tr.eq(yhat.round(), y_).float().mean()

    #=======backward pass=====================================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 50 == 0:
        print("[{}/{}], loss: {} acc: {}".format(e,
        num_epochs, np.round(loss.data[0], 3), np.round(acc.data[0], 3)))
