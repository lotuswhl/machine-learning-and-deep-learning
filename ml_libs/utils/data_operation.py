#__*__coding=utf-8__*__

import numpy as np

"""
数据计算相关工具
"""


def information_entropy(y):
    """
    计算信息熵,返回信息熵
    ---
    y:输入数据,通常是标签数据
    ---
    """
    log2 = lambda x:np.log(x)/np.log(2)
    entropy = 0.0
    unique_labels = np.lib.arraysetops.unique(y)
    for label in unique_labels:
        lc = len(y[y==label])
        p = lc/len(y)
        entropy += -p*log2(p)
    return entropy

def variance_cacl(X):
    """
    计算数据的方差
    """
    num_samples,num_features = np.shape(X)
    feature_means = np.ones_like(X)*np.mean(X,axis=0)
    
    variance = (1/num_samples)*np.dot((X-feature_means).T,(X-feature_means))
    return variance