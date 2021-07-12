import scipy.sparse as sp
import pandas as pd
import re
import numpy as np
import tensorflow as tf
# import tensorflow.keras.layers as Layers
import sys
import argparse
from time import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run Model.")
    parser.add_argument('--weights_path', nargs='?', default='', help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='', help='Project path.')

    parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
    parser.add_argument('--is_norm', type=int, default=1, help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--layer_size', nargs='?', default='[64, 64, 64, 64]')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]', help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.01,)

    parser.add_argument('--model_type', nargs='?', default='lightgcn',
                        help='Specify the name of model (lightgcn).')
    parser.add_argument('--adj_type', nargs='?', default='pre',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='lightgcn',
                        help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[20]', help='Top k(s) recommend')
    parser.add_argument('--save_flag', type=int, default=0, help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--report', type=int, default=0, help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    return parser.parse_args()

class GCN(object):
    def __init__(self, data_config, adj):
        super(GCN, self).__init__()
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        # self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        print(self.n_users, self.n_items)
        print(self.alg_type)
        self.n_fold = 100
        # self.norm_adj = adj
        # self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        print(self.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose
        self.Ks = eval(args.Ks)
        self.node_dropout_flag = args.node_dropout_flag

        def log(self):
            with tf.name_scope('TRAIN_LOSS'):
                self.train_loss = tf.placeholder(tf.float32)
                tf.summary.scalar('train_loss', self.train_loss)
                self.train_mf_loss = tf.placeholder(tf.float32)
                tf.summary.scalar('train_mf_loss', self.train_mf_loss)
                self.train_emb_loss = tf.placeholder(tf.float32)
                tf.summary.scalar('train_emb_loss', self.train_emb_loss)
                self.train_reg_loss = tf.placeholder(tf.float32)
                tf.summary.scalar('train_reg_loss', self.train_reg_loss)
                self.merged_train_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_LOSS'))
            with tf.name_scope('TRAIN_ACC'):
                self.train_rec_first = tf.placeholder(tf.float32)
                # record for top(Ks[0])
                tf.summary.scalar('train_rec_first', self.train_rec_first)
                self.train_rec_last = tf.placeholder(tf.float32)
                # record for top(Ks[-1])
                tf.summary.scalar('train_rec_last', self.train_rec_last)
                self.train_ndcg_first = tf.placeholder(tf.float32)
                tf.summary.scalar('train_ndcg_first', self.train_ndcg_first)
                self.train_ndcg_last = tf.placeholder(tf.float32)
                tf.summary.scalar('train_ndcg_last', self.train_ndcg_last)
            self.merged_train_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_ACC'))
            with tf.name_scope('TEST_LOSS'):
                self.test_loss = tf.placeholder(tf.float32)
                tf.summary.scalar('test_loss', self.test_loss)
                self.test_mf_loss = tf.placeholder(tf.float32)
                tf.summary.scalar('test_mf_loss', self.test_mf_loss)
                self.test_emb_loss = tf.placeholder(tf.float32)
                tf.summary.scalar('test_emb_loss', self.test_emb_loss)
                self.test_reg_loss = tf.placeholder(tf.float32)
                tf.summary.scalar('test_reg_loss', self.test_reg_loss)
            self.merged_test_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_LOSS'))
            with tf.name_scope('TEST_ACC'):
                self.test_rec_first = tf.placeholder(tf.float32)
                tf.summary.scalar('test_rec_first', self.test_rec_first)
                self.test_rec_last = tf.placeholder(tf.float32)
                tf.summary.scalar('test_rec_last', self.test_rec_last)
                self.test_ndcg_first = tf.placeholder(tf.float32)
                tf.summary.scalar('test_ndcg_first', self.test_ndcg_first)
                self.test_ndcg_last = tf.placeholder(tf.float32)
                tf.summary.scalar('test_ndcg_last', self.test_ndcg_last)
            self.merged_test_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_ACC'))
        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self.init_weights()
    # 初始化权重
    def init_weights(self):
        all_weights = dict()
        initializer = tf.random_normal_initializer(stddev=0.01)  # tf.contrib.layers.xavier_initializer()
        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
        self.weight_size_list = [self.emb_dim] + self.weight_size
        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)
            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)
            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)
        print("权重初始化完成")
        return all_weights

    def build(self):

    def create_ngcf_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            sum_embeddings = tf.nn.leaky_relu(tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])



            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])
            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            # ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

class loading():
    def __init__(self, path):
        self.file_path = path
        # # 获取电影信息
        # labels = ['movide_id', 'movie_title',  'video_release_date', 'NaN', 'IMDb_URL',
        #          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy ', 'Crime',
        #         'Documentary', 'Drama', 'Fantasy', 'Film - Noir', 'Horror', 'Musical', 'Mystery',
        #         'Romance', 'Sci - Fi', 'Thriller', 'War', 'Western']
        # self.item = pd.read_csv(path + 'u.item', sep="|", engine='python', names=labels)
        # self.item.drop(['NaN'], axis=1, inplace=True)
        # self.item['movie_title'], self.item['release_date'] = self.item['movie_title'].str.split('(', 1).str
        # self.item['release_date'] = self.item['release_date'].str.split(')', expand= True)
        # self.item = self.item[['movide_id', 'movie_title', 'release_date', 'video_release_date',
        #                       'IMDb_URL','unknown', 'Action', 'Adventure', 'Animation', 'Children\'s',
        #                       'Comedy ', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film - Noir',
        #                       'Horror', 'Musical', 'Mystery', 'Romance', 'Sci - Fi', 'Thriller', 'War',
        #                       'Western']]
        # # 获取用户信息
        # labels = ['user_id', 'age', 'gender', 'occupation', 'code']
        # self.user = pd.read_csv(path + 'u.user', sep="|", engine='python', names=labels)
        # self.occupation = pd.read_csv(path + 'u.occupation', engine='python', names=['occupation'])
        #
        # # print(self.occupation)
        # print(self.user.head())
        # print(self.item.head())

        self.ratings = pd.read_csv(path + 'u.data', sep="\t", engine='python', iterator=True, names=['user_id', 'movie_id', 'rating', 'timestamp'])
        self.ratings = self.ratings.get_chunk(5)
        self.ratings['avg_score'] = self.ratings.groupby(by='user_id')['rating'].transform('mean')
        self.user_num, self.item_num = self.ratings['user_id'].max() + 1, self.ratings['movie_id'].max() + 1
        print(self.user_num, self.item_num)
        self.train = self.ratings.sample(frac=0.8)
        # self.train = self.train.reset_index()
        self.test = self.ratings.drop(labels=self.train.index)
        # self.test = self.ratings.drop(labels=)
        print(self.train)
        print(self.test)
        self.R = sp.dok_matrix((self.user_num, self.item_num), dtype= np.float32 )
        self.ratings_matrix = sp.dok_matrix((self.user_num, self.item_num), dtype=np.float32)
        print(self.R.shape)
        for i,j in zip(self.train['user_id'], self.train['movie_id']):
            tmp = self.train.loc[self.train['user_id'] == i,['movie_id', 'rating']]
            rating = tmp.loc[tmp['movie_id'] == j,'rating'].values[0]
            # print(i, j, rating)
            self.R[i, j] = 1
            # rating 转 ctr
            self.ratings_matrix[i, j] = rating
            # self.R[i, j] = rating
        self.create_adj_mat()

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        print("------------交互矩阵-----------")
        print(R)
        print("------------------------------")
        # 分块处理，避免内存爆照
        num = 5.0
        for i in range(int(num)):
            adj_mat[int(self.user_num * i / num):int(self.user_num * (i + 1.0) / num), self.user_num:] = R[int(self.user_num * i / 5.0):int(self.user_num * (i + 1.0) / 5)]
            adj_mat[self.user_num:, int(self.user_num * i / num):int(self.user_num * (i + 1.0) / num)] = R[int(self.user_num * i / 5.0):int(self.user_num * (i + 1.0) / 5)].T
        adj_mat = adj_mat.todok()
        # 生成邻接矩阵
        print("------------邻接矩阵-----------")
        print(adj_mat)
        print("------------------------------")

        t2 = time()

        def normalized_adj_single(adj):
            # 矩阵按行求和
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            # 无访问，INF用0替换
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            print(d_mat_inv)
            sys.exit()
            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            # 检查标准化邻接矩阵是否等于这个拉普拉斯矩阵
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        # 添加单位矩阵
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)


        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat_inv)
        print('generate pre adjacency matrix.')
        pre_adj_mat = norm_adj.tocsr()

        plain_adj, norm_adj, mean_adj, pre_adj = adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr(), pre_adj_mat.toscr()
        return plain_adj, norm_adj, mean_adj, pre_adj

if __name__ == '__main__':
    file_path = '../data/ml-100k/'
    # load = loading(file_path)
    args = parse_args()
    data_config = {'n_users':1 , 'n_items': 1}
    model = GCN(data_config, adj='')
