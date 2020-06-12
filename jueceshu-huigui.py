# -*- coding: utf-8 -*-
"""
@author: Li Xianyang
"""
data=[]
with open('hemophilia.txt' ,'r',encoding='utf-8') as f:
    for line in f:
        if line.startswith('hiv'):
            continue
        data.append(list(map(float,line.split())))
        # print(line)
import numpy as np
import  pandas as pd
data=np.array(data)
df=pd.DataFrame(data,columns=['hiv', 'factor', 'year' ,'age' ,'py' ,'deaths'])
df.describe()


df['py']=(df['py']-df['py'].mean())/df['py'].std()
df['py'].describe()


data=np.array(df)
np.random.shuffle(data)
train_data = data[:int(data.shape[0] * 0.8)]
# print(train_data.dtype)
test_data = data[int(data.shape[0] * 0.8):]
print(f'train_data.shape: {train_data.shape}, test_data.shape: {test_data.shape}')

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
mlp= MLPRegressor()
print(mlp)
mlp.fit(train_data[:,:-1],train_data[:,-1])
mlp_y_predict=mlp.predict(train_data[:,:-1])
train_loss = mean_squared_error(train_data[:,-1], mlp_y_predict)
mlp_y_predict=mlp.predict(test_data[:,:-1])
test_loss = mean_squared_error(test_data[:,-1], mlp_y_predict)
print(f'the mse on the train data :{train_loss}, on the test data: {test_loss}')


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
dst= DecisionTreeRegressor()
print(dst)
dst.fit(train_data[:,:-1],train_data[:,-1])
dst_y_predict=dst.predict(train_data[:,:-1])
train_loss = mean_squared_error(train_data[:,-1], dst_y_predict)
dst_y_predict=dst.predict(test_data[:,:-1])
test_loss = mean_squared_error(test_data[:,-1], dst_y_predict)
print(dst_y_predict)
print(test_data[:,-1])
print(f'the mse on the train data :{train_loss}, on the test data: {test_loss}')