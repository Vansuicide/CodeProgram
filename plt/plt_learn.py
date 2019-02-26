# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
import numpy as np

# 简单应用
# x = np.linspace(-1, 1, 50)
# y = 2*x + 1
# plt.figure()
# plt.plot(x, y)
# plt.show()

# 简单的线条
x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2
# plt.figure()
# plt.plot(x, y1)
# plt.show()
plt.figure(num=3, figsize=(8, 5))

# legend图例
l1 = plt.plot(x, y1, label='linear line')
l2 = plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--', label='square line')
plt.legend(loc='upper right')  # 右上角

plt.xlim(-1, 2)
plt.ylim(-2, 3)
# plt.yticks([-2, -1.8, -1, 1.22, 3], [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
new_ticks = np.linspace(-1, 2, 5)  # x轴坐标范围[-1, 2]，个数5个
plt.xticks(new_ticks)
plt.yticks([-2, -1.8, 1.22, 3], ['$really\ bad$', '$bad$', '$normal$', '$good$', '$really\ good$'])
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))  # 横坐标位置修改
ax.spines['left'].set_position(('data', 0))  # 横坐标显示左半部分
ax.yaxis.set_ticks_position('left')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



