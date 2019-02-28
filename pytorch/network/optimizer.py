# -*-coding:utf-8-*-

import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(x.size()))


# plot data
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# put dataset into torch dataset
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(1, 20)
        self.predict = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


if __name__ == '__main__':
    # different net
    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    # different optimizer
    opt_SGD = optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam = optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = nn.MSELoss()
    loss_his = [[], [], [], []]  # record loss

    # training
    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        for step, (batch_x, batch_y) in enumerate(loader):
            for net, opt, l_his in zip(nets, optimizers, loss_his):
                output = net(batch_x)
                loss = loss_func(output, batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy())

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(loss_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()





