# -*- coding:utf-8 -*-

import numpy as np
from math import log

def calc_shannon_ent(category_list):
    label_count = {}
    num = len(category_list)
    for i in range(num):
        try:
            label_count[category_list[i]] += 1
        except KeyError:
            label_count[category_list[i]] = 1
    shannon_ent = 0.
    for k in label_count:
        prob = float(label_count[k]) / num
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent
def split_data(feature_matrix, category_list, feature_index, value):

    # feature_matrix = np.array(feature_matrix)
    ret_index = np.where(feature_matrix[:,feature_index] == value)[0]
    ret_category_list = [category_list[i] for i in ret_index]
    return ret_category_list
def choose_best_feature(feature_matrix, category_list):
    IG = []
    feature_num = len(feature_matrix[0])
    data_num = len(category_list)
    print(feature_num)
    print(data_num)
    base_shannon_ent = calc_shannon_ent(category_list=category_list)
    best_info_gain = 0
    best_feature_index = -1
    for f in range(feature_num):
        uni_value_list = set(feature_matrix[:,f])
        new_shannon_ent = 0.
        for value in uni_value_list:
            sub_cate_list = split_data(feature_matrix=feature_matrix, category_list=category_list, feature_index=f, value=value)
            prob = float(len(sub_cate_list)) / data_num
            new_shannon_ent += prob * calc_shannon_ent(sub_cate_list)
        info_gain = base_shannon_ent - new_shannon_ent
        f1=f+1
        print ('初始信息熵为：', base_shannon_ent, '按照特征%i 分类后的信息熵为：' % f1, new_shannon_ent, '信息增益:', info_gain)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature_index = f
        IG.append(info_gain)
    best_10_index = []
    tmp = sorted(IG,reverse=True)
    for i in range(10):
        best_10_index.append(IG.index(tmp[i])+1)

    return best_feature_index+1,best_10_index


 