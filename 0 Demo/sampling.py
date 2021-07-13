import numpy as np
import random
import sys

def maxk(arraylist,k):
    '''
    前k个的索引
    '''
    maxlist=[]
    maxlist_id= list(range(0,k+1))# 去掉自身首元素
    m=[maxlist,maxlist_id]
    for i in maxlist_id:
        maxlist.append(arraylist[i])
    for i in range(k,len(arraylist)):#对目标数组之后的数字
        if arraylist[i]>min(maxlist):
            mm=maxlist.index(min(maxlist))
            del m[0][mm]
            del m[1][mm]
            m[0].append(arraylist[i])
            m[1].append(i)
    return maxlist_id[1:]

def sampling(src_nodes, sample_num, neighbor_table):
    """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
    某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点

    Arguments:
        src_nodes {list, ndarray} -- 源节点列表
        sample_num {int} -- 需要采样的节点数
        neighbor_table {dict} -- 节点到其邻居节点的映射表

    Returns:
        np.ndarray -- 采样结果构成的列表
    """
    results = []
    for node in src_nodes:
        # 从节点的邻居中进行有放回地进行采样
        neighbor_rank = list(neighbor_table.todense().A[node, :])
        neighbors = maxk(neighbor_rank, sample_num)
        results = results + neighbors
    return results


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """根据源节点进行多阶采样

    Arguments:
        src_nodes {list, np.ndarray} -- 源节点id
        sample_nums {list of int} -- 每一阶需要采样的个数
        neighbor_table {dict} -- 节点到其邻居节点的映射

    Returns:
        [list of ndarray] -- 每一阶采样的结果
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result