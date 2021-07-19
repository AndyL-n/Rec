import time
import os
from collections import defaultdict
import math
import os
import random
import time
import math
import scipy.sparse as sp
import numpy as np
import sys
import re

class CF():
    # 初始化参数
    def __init__(self):
        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}
        self.train = []
        self.test = []
        # user和item集合
        self.user_set = set()
        self.item_set = set()
        self.n_users, self.n_items = 0, 0
        # user和item初始embedding
        # [1 + self.n_users * 879]
        self.user_emb = [0]
        # [1 + self.n_items * 91]
        self.item_emb = [0]
        # 用户相似度矩阵
        self.movie_sim_matrix = {}
        self.user_sim_matrix = {}
        # self.movie_popular = {}
        # self.movie_count = 0
        self.user_adj = []
        self.item_adj = []

    def load(self):
        self.get_dataset()
        self.item_similarity()
        self.user_similarity()
        sp.save_npz('user_adj', self.user_adj)
        sp.save_npz('item_adj', self.item_adj)
        # self.user_adj = sp.load_npz('user_adj.npz')
        # self.item_adj = sp.load_npz('item_adj.npz')
        self.item_embedding()
        self.user_embedding()
        return self.user_adj, self.item_adj,self.train,self.test,self.user_emb,self.item_emb
    # 读文件得到“用户-电影”数据
    def get_dataset(self, pivot=0.8):
        filename = '../data/ml-100k/u.data'
        trainSet_len = 0
        testSet_len = 0
        count = 0
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split('\t')
            self.user_set.add(user)
            self.item_set.add(movie)
            if (count % 10 < pivot * 10 ):
                self.trainSet.setdefault(user, {})
                self.trainSet[user][movie] = rating
                trainSet_len += 1
                self.train.append([int(user), int(movie), int(rating)])
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = rating
                testSet_len += 1
                self.test.append([int(user), int(movie), int(rating)])
            count += 1
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)
        self.n_users, self.n_items = len(self.user_set), len(self.item_set)
        self.user_adj = sp.dok_matrix((self.n_users + 1, self.n_users + 1), dtype=np.float32)
        self.user_adj = self.user_adj.tolil()
        self.item_adj = sp.dok_matrix((self.n_items + 1, self.n_items + 1), dtype=np.float32)
        self.item_adj = self.item_adj.tolil()
        print(self.user_adj.shape, self.item_adj.shape)

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
                self.item_adj[int(i), int(j)] = self.movie_sim_matrix[i][j]
                self.item_adj[int(j), int(i)] = self.movie_sim_matrix[i][j]
        self.item_adj += sp.eye(self.item_adj.shape[0])
        self.item_adj = self.item_adj.tocsr()

    """UserCF-LIF"""
    def user_similarity(self):
        """建立用户的协同过滤矩阵"""
        # 建立用户倒排表
        item_user = dict()
        for user, items in self.trainSet.items():
            for item in items:
                item_user.setdefault(item, set())
                item_user[item].add(user)

        # 建立用户协同过滤矩阵
        N = defaultdict(int)  # 记录用户购买商品数
        for item, users in item_user.items():
            for u in users:
                N[u] += 1
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, defaultdict(int))
                    self.user_sim_matrix[u][v] += 1. / math.log(1 + len(item_user[item]))


        # 计算相关度
        for u, related_users in self.user_sim_matrix.items():
            for v, con_items_count in related_users.items():
                self.user_sim_matrix[u][v] = con_items_count / math.sqrt(N[u] * N[v])
                self.user_adj[int(u), int(v)] = self.user_sim_matrix[u][v]
                self.user_adj[int(v), int(u)] = self.user_sim_matrix[u][v]

        self.user_adj += sp.eye(self.user_adj.shape[0])
        self.user_adj = self.user_adj.tocsr()

    """User_embedding"""
    def user_embedding(self):
        filename = '../data/ml-100k/u.user'
        ages, genders, occupations, addresses = [], [], [], []
        for line in self.load_file(filename):
            age, gender, occupation, address = line.split('|')[1:]
            ages.append(int(age))
            genders.append(gender)
            occupations.append(occupation)
            addresses.append(address)

        age, gender, occupation, address = list(set(ages)), list(set(genders)), list(set(occupations)), list(set(addresses))
        dim = len(age)+len(gender)+len(occupation)+len(address)
        print(len(age), len(gender), len(occupation), len(address))
        print(dim)
        for i in range(self.n_users):
            # print(ages[i],genders[i],occupations[i],addresses[i])
            one_hot = [0 for _ in range(dim)]
            one_hot[age.index(ages[i])] = 1
            one_hot[gender.index(genders[i]) + len(age)] = 1
            one_hot[occupation.index(occupations[i]) + len(age) + len(gender)] = 1
            one_hot[address.index(addresses[i]) + len(age) + len(gender) + len(occupation)] = 1
            one_hot = np.array(one_hot).astype(np.float).tolist()
            self.user_emb.append(one_hot)
        # self.user_emb = np.array(self.user_emb, dtype=float)

    """Item_embedding"""
    def item_embedding(self):
        filename = '../data/ml-100k/u.item'
        years = []
        with open(filename, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                line = line.split('|')
                one_hot = line[5:]
                one_hot[-1] = one_hot[-1][0]
                one_hot = [int(i) for i in one_hot]
                self.item_emb.append(one_hot)
                line = line[2].split('-')
                if len(line) == 3:
                    year = int(line[2])
                else:
                    year = 0
                years.append(year)
            year = list(set(years))
            # print(self.item_emb)
            for i in range(1,self.n_items+1):
                one_hot = [0.0 for _ in range(len(year))]
                one_hot[year.index(years[i-1])] = 1.0
                self.item_emb[i] = one_hot + self.item_emb[i]
                self.item_emb[i] = np.array(self.item_emb[i]).astype(np.float).tolist()
            # self.item_emb = np.array(self.item_emb, dtype=float)

if __name__ == '__main__':
    t1 = time.time()

    data = CF()
    print(data.train[:8])
    # print(len(data.item_emb))
    # data.item_similarity()
    # print(data.item_adj.shape)
    # print(data.item_adj.todense())
    # data.user_similarity()
    # print(data.user_adj.shape)
    # print(data.user_adj.todense())
    # print(data.trainSet)
    print("time:{}".format(time.time() - t1))
