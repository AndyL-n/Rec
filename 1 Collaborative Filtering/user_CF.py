# coding = utf-8
# 基于用户的协同过滤推荐算法实现

import random
import os
import time
import math
from operator import itemgetter
from utils import metric


class UserBasedCF():
    # 初始化相关参数
    def __init__(self):
        # 找到与目标用户兴趣相似的20个用户，为其推荐10部电影
        self.n_sim_user = 20
        self.n_rec_movie = 10

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}
        # user和item集合
        self.user_set = set()
        self.item_set = set()

        # 用户相似度矩阵
        self.user_sim_matrix = {}
        self.movie_count = 0

        print('Similar user number = %d' % self.n_sim_user)
        print('Recommneded movie number = %d' % self.n_rec_movie)

    def get_dataset(self, filename, pivot=0.75):
        """
        读文件得到“用户-电影”数据
        :param filename:
        :param pivot:
        :return:
        """
        trainSet_len = 0
        testSet_len = 0
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split('\t')
            self.user_set.add(user)
            self.item_set.add(movie)
            if random.random() < pivot:
                self.trainSet.setdefault(user, {})
                self.trainSet[user][movie] = rating
                trainSet_len += 1
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = rating
                testSet_len += 1

        print('Split trainingSet and testSet success!')
        print(self.trainSet['186'])
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)


    # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)


    def calc_user_sim(self):
        """
        计算用户之间的相似度
        :return:
        """
        # 构建“电影-用户”倒排索引
        # key = movieID, value = list of userIDs who have seen this movie
        print('Building movie-user table ...')
        movie_user = {}
        for user, movies in self.trainSet.items():
            for movie in movies:
                if movie not in movie_user:
                    movie_user[movie] = set()
                movie_user[movie].add(user)
        print('Build movie-user table success!')

        self.movie_count = len(movie_user)
        print('Total movie number = %d' % self.movie_count)

        print('Build user co-rated movies matrix ...')
        for movie, users in movie_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    self.user_sim_matrix[u][v] += 1
        print('Build user co-rated movies matrix success!')
        # print(self.user_sim_matrix)
        # 计算相似性
        print('Calculating user similarity matrix ...')
        for u, related_users in self.user_sim_matrix.items():
            for v, count in related_users.items():
                self.user_sim_matrix[u][v] = count / math.sqrt(len(self.trainSet[u]) * len(self.trainSet[v]))
        print('Calculate user similarity matrix success!')

    def recommend(self, user):
        """
        针对目标用户U，找到其最相似的K个用户，产生N个推荐
        :param user:
        :return:
        """
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainSet[user]

        # v=similar user, wuv=similar factor
        for v, wuv in sorted(self.user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0:K]:
            for movie, rvi in self.trainSet[v].items():
                if movie in watched_movies:
                    continue
                rank.setdefault(movie, 0)
                rank[movie] += wuv * float(rvi)  # rvi评分
                # rank[movie] += wuv
        # print(sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N])
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]


    def evaluate(self):
        """
        产生推荐并通过准确率、召回率和覆盖率等进行评估
        :return:
        """
        print("Evaluation start ...")
        test_user_items = dict()
        # 推荐
        recommed_dict = dict()
        for user, v in self.testSet.items():
            recommed = self.recommend(user)
            recommed_dict.setdefault(user, list())
            for item, score in recommed:
                recommed_dict[user].append(item)
            test_user_items[user] = list(v.keys())

        item_popularity = dict()
        for user, v in self.trainSet.items():
            items = v.keys()
            for item in items:
                if item in item_popularity:
                    item_popularity[item] += 1
                else:
                    item_popularity.setdefault(item, 1)

        # 评分
        precision = metric.precision(recommed_dict, test_user_items)
        recall = metric.recall(recommed_dict, test_user_items)
        coverage = metric.coverage(recommed_dict, self.item_set)
        popularity = metric.popularity(item_popularity, recommed_dict)
        print("precision:{:.4f}, recall:{:.4f}, coverage:{:.4f}, popularity:{:.4f}".format(precision, recall, coverage,
                                                                                           popularity))
        # hit = metric.hit(recommed_dict, test_user_items)
        # print(hit)

if __name__ == '__main__':
    t1 = time.time()
    rating_file = '../data/ml-100k/u.data'
    userCF = UserBasedCF()
    userCF.get_dataset(rating_file)
    userCF.calc_user_sim()
    userCF.evaluate()
    print("time:{}".format(time.time() - t1))
    # precision:0.3519, recall:0.1344, coverage:0.2004, popularity:5.3571
