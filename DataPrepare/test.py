###############
# reshape
import numpy as np
a = np.array(range(0,24))
print(a)
b = a.reshape((2,3,4))
print(b)
c = b.reshape((2,3*4))
print(c)

# ###############
# #L1Loss
# from numpy import float32
# import torch
# import torch.nn.functional as F
# import numpy as np
# from xarray import Coordinate 

# loss = F.l1_loss
# O1 = np.array([[1,2,3,4,5],
# [1,2,3,4,5]],dtype=float)
# F1 = np.array([[1,2,3,4,5],
# [2,3,4,5,0]],dtype=float)

# coor = O1.nonzero()
# O1 = O1[coor]
# F1 = F1[coor]
# coor = F1.nonzero()
# O1 = O1[coor]
# F1 = F1[coor]

# input = torch.tensor(O1,dtype=float  )
# target = torch.tensor(F1,dtype=float)
# print(input)
# print(target)
# output = loss(input, target)
# print(output)


###############
#正则表达式
# import re
# pattern = re.compile(r'Used\s+:\s+([0-9]+)\s+MiB')
# str = u'''==============NVSMI LOG==============

# Timestamp                                 : Fri Jun 10 12:48:16 2022
# Driver Version                            : 510.73.08
# CUDA Version                              : 11.6

# Attached GPUs                             : 1
# GPU 00000000:01:00.0
#     FB Memory Usage
#         Total                             : 24576 MiB
#         Reserved                          : 309 MiB
#         Used                              : 4951 MiB
#         Free                              : 19315 MiB
#     BAR1 Memory Usage
#         Total                             : 256 MiB
#         Used                              : 5 MiB
#         Free                              : 251 MiB
# '''
# result = pattern.match(str)
# print(result)




#######################
#2D正态分布

# # np.set_printoptions(precision=2, suppress=True)


# # Importing Numpy package
# import numpy as np
# np.set_printoptions(precision=2, suppress=True)
 
# # sigma(standard deviation) and muu(mean) are thre parameters of gaussian
 
 
# def gaussian_filter(kernel_size, sigma=1, muu=0, centerX=0, centerY=0,centerValue=1, point=(0,0)):
 
#     # Initializing value of x,y as grid of kernel size
#     # in the range of kernel size
 
#     x, y = np.meshgrid(np.linspace(-2, 2, kernel_size),
#                        np.linspace(-2, 2, kernel_size))

#     print(f'{calcGaussian(sigma, muu, centerX, centerY, centerValue, point[0],point[1]):.2f}')

#     gauss = calcGaussian(sigma, muu, centerX, centerY, centerValue, x, y)
#     return gauss

# def calcGaussian(sigma, muu, centerX, centerY, centerValue, pointX, pointY):
#     dst = np.sqrt((pointX-centerX)**2+(pointY-centerY)**2)
#     normalV = 1/(2.0 * np.pi * sigma**2) * centerValue
#     gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normalV
#     return gauss
 
# kernel_size = 9
# centerX=np.array( [1,1.5])
# centerY=np.array([1,1])
# centerValue=np.array([100,200])
# print(calcGaussian(0.25,0,centerX,centerY,centerValue,0.5,0.5))
# print(calcGaussian(0.5,0,centerX,centerY,centerValue,0.5,0.5))
# # print( gaussian_filter(kernel_size,sigma=0.25, centerX=1,centerY=1,centerValue=100, point=(0.5,0.5)))
