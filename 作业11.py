# -*- coding: utf-8 -*-
"""
@author: Li Xianyang
"""

import pandas as pd
df = pd.read_excel('偷税漏税.xls')
df.head()


df = df.drop(['纳税人编号'], axis=1)
variables_need_progress=['销售类型','销售模式','输出']
mymap={}
for col in variables_need_progress:
    names=df[col].unique()
    for i,name in enumerate(names):
        mymap[name]=i

for col in variables_need_progress:
    for row in range(df.shape[0]):
        df.loc[row,col]=mymap[ df.loc[row,col]]
    df[col]=df[col].astype('int')
df.head()

df.describe()


others=list(df.columns)
for x in variables_need_progress:
    others.remove(x)
df.loc[:,others]=(df.loc[:,others]-df.loc[:,others].mean())/df.loc[:,others].std()
df.describe()

from sklearn.model_selection import train_test_split
import numpy as np
data=np.array(df.iloc[:,:-1])
target=np.array(df.iloc[:,-1])
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2)
print('x_train.shape',x_train.shape,'x_test.shape',x_test.shape)

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(max_iter=1000)
mlp.fit(x_train,y_train)
mlppre=mlp.predict(x_test)
print('{:*^40}'.format('神经网络评估结果'))
print(classification_report(y_test,mlppre))


dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
dtcpre=dtc.predict(x_test)
print('{:*^40}'.format('决策树评估结果'))
print(classification_report(y_test,dtcpre))
