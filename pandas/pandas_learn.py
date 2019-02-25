# -*-coding:utf-8-*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建数据
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])

# print(df)

# 根据位置设置 loc 和 iloc
df.iloc[2, 2] = 1111
df.loc['20130101', 'B'] = 2222

# print(df)

# 根据条件设置
df.B[df.A > 4] = 0

# print(df)

# 按行或列设置
df['F'] = np.nan

# print(df)

# Series可视化
# data = pd.Series(np.random.randn(1000), index=np.arange(1000))
# data.cumsum()
# data.plot()
# plt.show()

# DataFrame可视化
data = pd.DataFrame(
    np.random.randn(1000, 4),
    index=np.arange(1000),
    columns=list("ABCD")
)
data.cumsum()
data.plot()
plt.show()


