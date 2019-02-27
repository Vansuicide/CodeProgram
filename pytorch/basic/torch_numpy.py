# -*-coding:utf-8-*-

import torch
import numpy as np

# convert numpy to tensor or vise versa
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array:', np_data,
    '\ntorch tensor:', torch_data,
    '\ntensor to array:', tensor2array
)


# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32-bit floating point
print(
    '\nabs',
    '\nnumpy: ', np.abs(data),
    '\ntorch: ', torch.abs(tensor)
)

# sin
print(
    '\nsin',
    '\nnumpy: ', np.sin(data),
    '\ntorch: ', torch.sin(tensor)
)

# mean
print(
    '\nmean',
    '\nnumpy: ', np.mean(data),
    '\ntorch: ', torch.mean(tensor)
)

# matrix multiplication
data = [[1, 2], [3, 4]]
tensor1 = torch.FloatTensor(data)
# correct method
print(
    '\nmatrix multiplication',
    '\nnumpy: ', np.matmul(data, data),
    '\ntorch: ', torch.mm(tensor1, tensor1)
)
# incorrect method
data = np.array(data)
print(
    '\nmatrix multiplication',
    '\nnumpy: ', data.dot(data),
    '\ntorch: ', tensor1.dot(tensor1)  # RuntimeError: dot: Expected 1-D argument self, but got 2-D
)
