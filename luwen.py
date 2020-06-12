from sklearn import datasets
all_data=datasets.fetch_california_housing()
x=all_data['data']
y=all_data['target']
import pandas as pd
df=pd.DataFrame(x)
print(df.shape)