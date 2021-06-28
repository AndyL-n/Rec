import time
import os
from collections import defaultdict
import math
from item_CF import ItemBasedCF

class ItemIUF(ItemBasedCF):
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

if __name__ == '__main__':
    t1 = time.time()
    rating_file = '../data/ml-100k/u.data'
    item_iuf = ItemIUF()
    item_iuf.get_dataset(rating_file)
    item_iuf.item_similarity()
    item_iuf.evaluate()
    print("time:{}".format(time.time() - t1))
    # precision:0.4010, recall:0.1257, coverage:0.1100, popularity:5.4350
