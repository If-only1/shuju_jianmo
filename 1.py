from sklearn.datasets import load_boston
import pandas as pd
from torch import nn
import torch
import numpy as np

df = load_boston()
# print(df.data.shape)
# print(df.target.shape)

data = np.concatenate([df.data,df.target.reshape(-1,1)],axis=1).astype(np.float32)
# np.random.shuffle(data)
print('data.shape',data.shape)  # (506, 13)
train_data = data[:int(data.shape[0] * 0.8)]
# print(train_data.dtype)
test_data = data[int(data.shape[0] * 0.8):]
print(f'train_data.shape: {train_data.shape}, test_data.shape: {test_data.shape}')



from sklearn.neural_network import MLPRegressor

lr=MLPRegressor(max_iter=1000,verbose=True)
print(lr.fit(train_data[:,:-1],train_data[:,-1:]))
lr_y_predict=lr.predict(test_data[:,:-1])
from sklearn.metrics import mean_squared_error
score = mean_squared_error(test_data[:,-1:], lr_y_predict)
print(score)
exit()



feature_names = list(df.feature_names).append('TARGET')
data_df = pd.DataFrame(data, columns=feature_names)
data_df.describe()
exit()
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F


class Mydataset(Dataset):
    def __init__(self, data, feature_names, transform=None):
        super().__init__()
        self.data = data
        self.feature_names = feature_names
        self.tansform = transform
        if self.tansform:
            self.data = self.tansform(data).squeeze(0)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


import torchvision

train_dataset = Mydataset(train_data, feature_names, torchvision.transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

test_dataset = Mydataset(test_data, feature_names, torchvision.transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, drop_last=True)


class Mynet1(nn.Module):
    def __init__(self, in_nums, hide_nums, out_nums):
        super().__init__()

        self.fc1 = nn.Linear(in_nums, hide_nums)
        self.fc2 = nn.Linear(hide_nums, out_nums)
        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


model = Mynet1(13, 1000, 1)
from torch.optim import *
lr=0.001
print(list(model.parameters()))
# exit()
opt = Adam(model.parameters(), lr=lr)
opt_moment = SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = torch.nn.MSELoss()


def train(epochs, model, opt,dataloader,mode='train'):
    loss_list=[]
    if mode=='train':
        model.train()
    else:
        model.eval()
    for epoch in range(epochs):
        for data in dataloader:
            opt.zero_grad()
            x = data[:, :-1]
            y = data[:, -1:]
            with torch.set_grad_enabled(mode=='train'):
                pre = model(x)
                loss = criterion(pre, y)
                if mode=='Train':
                    loss.backward()
                    opt.step()
                print(f'epoch:{epoch}/{epochs}, loss on the {mode} set: {loss.item()} ')
                loss_list.append(loss.item())
    return loss_list


ans=train(1000, model, opt,train_dataloader,mode='train')
ans1=train(1, model, opt,test_dataloader,mode='test')

model=Mynet1(13,1000,1)
ans2=train(1000, model, opt_moment,train_dataloader,mode='train')
ans2=train(1, model, opt,test_dataloader,mode='test')
print(ans1,ans2)
