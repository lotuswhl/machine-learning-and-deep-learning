#!/usr/bin/python3
#coding=utf-8
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm

#  正态分布示例：连续

# fix random seed
np.random.seed(47)

X = np.arange(-5, 5, 0.001)

y = norm.pdf(X,0,1)

fig = plt.figure()

plt.plot(X,y)

plt.tight_layout()

plt.savefig("normal-distribution.png")


# 离散

numSamples = 40

X = np.random.normal(0, 1, numSamples)

y = np.zeros(numSamples)

fig = plt.figure(figsize=(8,4))

plt.scatter(X, y, color='r')

# 设置figure x轴的范围
plt.xlim((-4,4))

# 获取当前的坐标轴axes
frame = plt.gca()

frame.axes.get_yaxis().set_visible(False)

plt.savefig("normal-samples.png")


# 多元高斯示例

X = np.arange(-4, 4, 0.01)
y = np.arange(-4, 4, 0.01)

X,y = np.meshgrid(X,y)

import matplotlib.mlab as mlab

Z = mlab.bivariate_normal(X, y,1.0,1.0,0.0,0.0)


plt.figure()

# 绘制轮廓
contour = plt.contour(X,y,Z)

# 绘制标签
plt.clabel(contour,inline=2,fontsize=10)

numSamples = 240

mean = [0,0]
cov = [[1,0],[0,1]]

X,y = np.random.multivariate_normal(mean, cov, numSamples).T

plt.scatter(X, y,color='k')

plt.savefig("multi-variate normal-samples.png")
