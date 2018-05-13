#__*__coding=utf-8__*__

import numpy as np
from numpy.lib.arraysetops import unique
"""
数据处理相关工具
"""


def separate_data_on_feature_by_threshold(Xy,feature_index,threshold):
    """
    根据样本数据的特征以及特征的阈值来分割样本;并将分割后的数据转化为numpy数组,便于计算
    ---
    Parameters:
    Xy: 输入的样本的数据特征,可以是和样本标签拼接后的数据,第一个维度是样本大小
    feature_index: 特征索引 feature_index ,在此特征上进行分割
    threshold: 待分割的特征的阈值
    ---
    """
    if isinstance(threshold, (int,float)):
        separate_func = lambda sample:sample[feature_index]>=threshold
    else:
        separate_func = lambda sample:sample[feature_index]==threshold
    
    Xy_right= np.array([sample for sample in Xy if separate_func(sample)])
    Xy_left= np.array([sample for sample in Xy if not separate_func(sample)])
    return (Xy_left,Xy_right)

def calc_gini_group_score(group):
    unique_labels = unique(group)
    size = len(group)
    scores = 0.0
    if size==0:
        return 0
    for label in unique_labels:
        p = 0.0
        label_cnt=len(group[group==label])
        p = label_cnt/size
        scores += p*p
    return 1-scores




if __name__=="__main__":
    # test separate_data_on_feature_by_threshold
    # data = np.array([[1,2,3,4,5,6],
    #                  [3,4,5,7,8,9],
    #                  [12,3,2,3,4,4],
    #                  [3,3,323,43,2,4]])
    # print(separate_data_on_feature_by_threshold(data,2, 5))

    group = np.array([2,2,2,1])
    print(calc_gini_group_score(group))