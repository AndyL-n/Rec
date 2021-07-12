import time
import os
from collections import defaultdict
import math
import os
import random
import time
import math
from operator import itemgetter
from utils import metric

class ItemIUF():
    # 初始化参数
    def __init__(self):

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}
        # user和item集合
        self.user_set = set()
        self.item_set = set()
        # 用户相似度矩阵
        self.movie_sim_matrix = {}
        self.movie_popular = {}
        self.movie_count = 0


    # 读文件得到“用户-电影”数据
    def get_dataset(self, filename, pivot=1):
        trainSet_len = 0
        testSet_len = 0
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split('\t')
            self.user_set.add(user)
            self.item_set.add(movie)
            if (random.random() < pivot):
                self.trainSet.setdefault(user, {})
                self.trainSet[user][movie] = rating
                trainSet_len += 1
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = rating
                testSet_len += 1
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)

    # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
            #     if i == 0:  # 去掉文件第一行的title
            #         continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)

    """ItemCF-IUF"""
    def item_similarity(self):
        N = defaultdict(int)  # 每个物品的流行度

        # 统计同时购买商品的人数
        for _, items in self.trainSet.items():
            for i in items:
                self.movie_sim_matrix.setdefault(i, dict())
                # 统计商品的流行度
                N[i] += 1

                for j in items:
                    if i == j:
                        continue
                    self.movie_sim_matrix[i].setdefault(j, 0)
                    self.movie_sim_matrix[i][j] += 1. / math.log1p(len(items) * 1.)

        # 计算物品协同矩阵
        for i, related_items in self.movie_sim_matrix.items():
            for j, related_count in related_items.items():
                self.movie_sim_matrix[i][j] = related_count / math.sqrt(N[i] * N[j])
        print(len(self.movie_sim_matrix))

if __name__ == '__main__':
    t1 = time.time()
    rating_file = '../data/ml-100k/u.data'
    item_iuf = ItemIUF()
    item_iuf.get_dataset(rating_file)
    item_iuf.item_similarity()
    # item_iuf.evaluate()
    print("time:{}".format(time.time() - t1))
    # precision:0.4010, recall:0.1257, coverage:0.1100, popularity:5.4350
