from sklearn import datasets
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
all_data = datasets.fetch_california_housing(proprecessing=False)
feature_names=all_data.feature_names
target_names=all_data.target_names
DESCR=all_data.DESCR
# print(DESCR)
x = all_data['data']
y = all_data['target'].reshape(-1,1)
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
import torch
import torchvision
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset

train_data_tensor = torch.Tensor(train)
test_data_tensor = torch.Tensor(test)
print('train_data_tensor.size(),test_data_tensor.size(),',
      train_data_tensor.size(), test_data_tensor.size(), )
# train_dataset = Mydataset(train_data, feature_names, torchvision.transforms.ToTensor())
train_dataset = TensorDataset(train_data_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)


import torch
import luwen.mymodel as mod
# 通过实例化上面定义的类来构建我们的模型。
model = mod.Mlp(100)

# 构造损失函数和优化器。
# SGD构造函数中对model.parameters()的调用，
# 将包含模型的一部分，即两个nn.Linear模块的可学习参数。
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss(reduction='mean')


def train(model,train_dataloader,loss_fn,optimizer):
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

pd.to_pickle()

