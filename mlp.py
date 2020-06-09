from sklearn.datasets import load_boston
import pandas as pd
from torch import nn
import torch

df=load_boston()
data=df.data
feature_names=df.feature_names
print(data.shape)
data_df=pd.DataFrame(data,columns=feature_names)
print(data_df.head())
print(data_df.describe())

import torch.nn.functional as F

class Mynet1(nn.Module):
    def __init__(self,in_nums,hide_nums,out_nums):
        self.fc1=nn.Linear(in_nums,hide_nums)
        self.fc2=nn.Linear(hide_nums,out_nums)
        nn.init.xavier_normal(self.fc1.weight)
        nn.init.xavier_normal(self.fc2.weight)
    def forward(self,x):
        x=self.fc1(x)
        x=F.sigmoid(x)
        x=self.fc2(x)
        return x

from torch.optim import SGD
opt=SGD()