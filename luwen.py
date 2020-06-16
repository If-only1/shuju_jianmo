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



