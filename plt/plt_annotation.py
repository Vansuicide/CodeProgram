# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y = 2*x + 1

plt.figure(num=1, figsize=(8, 5))
plt.plot(x, y)
ax = plt.gca()
ax.spines['right'].set_color('none')  # 右边框
ax.spines['top'].set_color('none')  # 左边框
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

x0 = 1
y0 = 2 * x0 + 1
plt.plot([x0, x0], [0, y0], 'k--', linewidth=2.5)  # 前面两个都是x坐标，后面两个都是y坐标
# set dot styles
plt.scatter([x0, ], [y0, ], s=50, color='b')

# 添加注释, xycoords根据data位置来选位置，xytext和textcoords是对标注位置的描述和xy偏差值，arrowprops是箭头的一些设置
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3, rad=.2"))
# 添加文字，-3.7和3是文字的起始位置， fontdict是字体格式
plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size': 16, 'color': 'r'})
plt.show()



