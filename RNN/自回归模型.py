import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

T = 1000
time = torch.arange(1, T+1, dtype=torch.float32)
x = torch.sin(0.01*time) + torch.normal(0, 0.2, (T,))

plt.plot(time, x)
plt.show()
plt.close()

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(tau, 10), nn.ReLU(), nn.Linear(10, 1))
    def forward(self, input):
        return self.net(input)

tau = 4
lr = 1e-3
net = Net().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion = nn.MSELoss()


# 构造数据集
feat = torch.zeros((T-tau, tau))
label = x[tau:]
for i in range(tau):
    feat[:, i] = x[i:T-tau+i]


class myData(Dataset):
    def __init__(self, feat, label, train = True):
        super().__init__()
        self.train = train
        if train:
            self.feat = feat[:600]
            self.label = label[:600]
        else:
            self.feat = feat[600:]
            self.label = label[600:]
    def __getitem__(self, index):
        return self.feat[index], self.label[index]
    def __len__(self):
        return len(self.label)

train_data = myData(feat, label, True)
validate_data = myData(feat, label, False)

train_loader = DataLoader(train_data, 32, shuffle=True, num_workers=2)
validate_loader = DataLoader(validate_data, 32, shuffle=False, num_workers=2)

n_epoch = 15
for epoch in range(n_epoch):
    net.train()
    for X, y in train_loader:
        X = X.to('cuda')
        y = y.to('cuda').view(-1,)

        pred = net(X).view((-1,))
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'\rEpoch:{epoch+1} loss:{loss.item():>.6f}', end='')
    print('')

pred = net(feat.cuda()).view((-1,)).cpu()
plt.plot(range(len(x)), x, 'r--', label = 'data')
plt.plot(range(len(pred.detach())), pred.detach(), 'b--', label = 'pred')
plt.legend()
plt.show()
plt.close()

multistep_preds = torch.zeros(T)
multistep_preds[: 600 + tau] = x[: 600 + tau]
for i in range(600 + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)).cuda()).cpu()

pred = net(feat.cuda()).view((-1,)).cpu()
plt.plot(range(len(x)), x, 'r--', label = 'data')
plt.plot(range(len(pred.detach())), pred.detach(), 'g--', label = 'pred')
plt.plot(range(len(multistep_preds.detach())), multistep_preds.detach(), 'b--', label = 'mul_pred')
plt.legend()
plt.show()
plt.close()