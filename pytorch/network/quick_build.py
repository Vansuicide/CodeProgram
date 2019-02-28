# -*-coding:utf-8-*-

import torch
import torch.nn.functional as F
import torch.nn as nn


# replace following class code with an easy sequential network
class Net(nn.Module):
    def __init__(self, n_features, n_hiddens, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hiddens)
        self.predict = torch.nn.Linear(n_hiddens, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net1 = Net(1, 10, 1)

# easy and fast way to build your network
net2 = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
)

print(net1)

print(net2)

