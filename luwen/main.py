from sklearn import datasets
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
all_data = datasets.fetch_california_housing()
feature_names=all_data.feature_names
target_names=all_data.target_names
DESCR=all_data.DESCR
# print(DESCR)
x = all_data['data']
y = all_data['target'].reshape(-1,1)


#miao shu xing tongji
df = pd.DataFrame(np.concatenate([x,y],axis=1),columns=feature_names+target_names)
print(df.info())
print(df.describe())
dx=None
for i,name in enumerate(feature_names):
    plt.subplot(2,4,i+1)
    re=plt.boxplot(df[name],
                sym='o',  # 异常点形状
                vert=True,  # 是否垂直
                whis=1.5,  # IQR
                patch_artist=True,  # 上下四分位框是否填充
                meanline=False, showmeans=True,  # 是否有均值线及其形状
                showbox=True,  # 是否显示箱线
                showfliers=True,  # 是否显示异常值
                notch=False,  # 中间箱体是否缺口
                )
    if  dx is None:
        dx=re['fliers'][0].get_xdata()
    else:
        dx=np.concatenate([dx,re['fliers'][0].get_xdata()], axis=0)
    plt.title(name)

plt.show()
print(dx.shape)

def remove_filers_with_boxplot(data):
    p = data.boxplot(return_type='dict')
    for index,value in enumerate(data.columns):
        # 获取异常值
        fliers_value_list = p['fliers'][index].get_ydata()
        # 删除异常值
        for flier in fliers_value_list:
            data = data[data.loc[:,value] != flier]
    return data
print(df.shape)
df = remove_filers_with_boxplot(df)
print(df.shape)


#suiji daluan shuju  hua fen wei xunlianji  ceshiji

df = df.reset_index()
size=0.5
train_nums=int(df.shape[0]*size)
train = df.loc[0:train_nums]
test=df.loc[train_nums:]
print('train.shape',train.shape)
print('train.shape',test.shape)


#no nomalizition


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




import torch
import torchvision
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset

train_data_tensor = torch.Tensor(train_data)
test_data_tensor = torch.Tensor(test_data)
print('train_data_tensor.size(),test_data_tensor.size(),',
      train_data_tensor.size(), test_data_tensor.size(), )
# train_dataset = Mydataset(train_data, feature_names, torchvision.transforms.ToTensor())
train_dataset = TensorDataset(train_data_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)


import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        在构造函数中，我们实例化了两个nn.Linear模块，并将它们作为成员变量。
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        在前向传播的函数中，我们接收一个输入的张量，也必须返回一个输出张量。
        我们可以使用构造函数中定义的模块以及张量上的任意的（可微分的）操作。
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

# N是批大小； D_in 是输入维度；
# H 是隐藏层维度； D_out 是输出维度
N, D_in, H, D_out = 64, 13, 100, 1



# 通过实例化上面定义的类来构建我们的模型。
model = TwoLayerNet(D_in, H, D_out)

# 构造损失函数和优化器。
# SGD构造函数中对model.parameters()的调用，
# 将包含模型的一部分，即两个nn.Linear模块的可学习参数。
loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for t in range(200):
    # 前向传播：通过向模型传递x计算预测值y
    for data in train_dataloader:
        data=data[0]
        x=data[:,:-1]
        y=data[:,-1:]
        y_pred = model(x)

        #计算并输出loss
        loss = loss_fn(y_pred, y)
        print(t, loss.item())

        # 清零梯度，反向传播，更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
pre=model(test_data_tensor[:,:-1]).detach().numpy()
print(pre)

# test_loss=((pre-test_data[:,-1:])**2).mean()
test_loss=mean_squared_error(pre,test_data[:,-1:])
print(test_loss)


