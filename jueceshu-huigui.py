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
print(df)