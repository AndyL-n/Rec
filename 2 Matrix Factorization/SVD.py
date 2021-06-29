import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import sys

class SVD():
    def __init__(self, K):
        # 奇异值个数
        self.K = K
        self.trainset = {}
        self.testset = {}
        self.userList = []
        self.itemList = []
        # 原始矩阵
        self.M = None
        # 分解所得三个矩阵
        self.u = None
        self.i = None
        self.v = None

    def get_dataset(self, path):
        data = pd.read_csv(path)
        self.testset = data.sample(frac=0.2, axis=0)
        self.trainset = data.drop(index=self.testset.index.values.tolist(), axis=0)

    def getMatrix(self):
        userSet, itemSet = set(), set()
        for d in self.trainset.values:
            # print(d[0])
            d = list(map(int, d[0].split('\t')))
            # d = np.char.split(d[0],sep='\t')
            userSet.add(int(d[0]))
            itemSet.add(int(d[1]))

        self.userList = list(userSet)
        self.itemList = list(itemSet)

        self.M = pd.DataFrame(0, index=self.userList, columns=self.itemList, dtype=float)
        for d in tqdm(self.trainset.values):
            d = list(map(int, d[0].split('\t')))
            self.M[d[1]][d[0]] = d[2]

    def svd(self):
        self.u, self.i, self.v = np.linalg.svd(self.M)
        self.u, self.i, self.v = self.u[:, 0:self.K], np.diag(self.i[0:self.K]), self.v[0:self.K, :]

    def predict(self, user_index, item_index):
        return float(self.u[user_index].dot(self.i).dot(self.v.T[item_index].T))

    def getPredicts(self, Set):
        y_true, y_hat = [], []

        for d in tqdm(Set.values):
            d = list(map(int, d[0].split('\t')))
            user = int(d[0])
            item = int(d[1])
            if user in self.userList and item in self.itemList:
                user_index = self.userList.index(user)
                item_index = self.itemList.index(item)
                y_true.append(d[2])
                y_hat.append(self.predict(user_index, item_index))
        return y_true, y_hat



    def RMSE(self, a, b):
        return (np.average((np.array(a) - np.array(b)) ** 2)) ** 0.5
if __name__ == '__main__':
    t1 = time.time()
    rating_file = '../data/ml-100k/u.data'
    svd = SVD(200)
    svd.get_dataset(rating_file)
    svd.getMatrix()
    svd.svd()
    train_y_true, train_y_hat = svd.getPredicts(svd.trainset)
    test_y_true, test_y_hat = svd.getPredicts(svd.testset)
    print(svd.RMSE(train_y_true, train_y_hat))
    print(svd.RMSE(test_y_true, test_y_hat))
