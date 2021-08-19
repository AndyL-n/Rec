"""
author: L
date: 2021/8/19 14:54
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--weights_path', nargs='?', default='',help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='Data/',help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',help='Project path.')
    parser.add_argument('--dataset', nargs='?', default='gowalla',help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
    parser.add_argument('--pretrain', type=int, default=0,help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,help='Interval of evaluation.')
    parser.add_argument('--is_norm', type=int, default=1,help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000,help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=64,help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64, 64, 64, 64]',help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.01,help='Learning rate.')
    parser.add_argument('--model_type', nargs='?', default='lightgcn',help='Specify the name of model (lightgcn).')
    parser.add_argument('--adj_type', nargs='?', default='pre',help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='lightgcn',help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')
    parser.add_argument('--gpu_id', type=int, default=0,help='0 for NAIS_prod, 1 for NAIS_concat')
    parser.add_argument('--node_dropout_flag', type=int, default=0,help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--Ks', nargs='?', default='[20]',help='Top k(s) recommend')
    parser.add_argument('--save_flag', type=int, default=0,help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--test_flag', nargs='?', default='part',help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--report', type=int, default=0,help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    return parser.parse_args()

class LightGCNLayer(Layer):
    def __init__(self):
        """
        GCN Part
        """

    def build(self, input_shape):

        self.weights =

    def call(self, inputs, **kwargs):
        return

class LightGCN(Model):
    def __init__(self,arg,data_config):
        """
        LightGCN Part
        """


        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.norm_adj = data_config['norm_adj']

        self.n_fold = 100
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.log_dir = self.create_model_str()
        self.verbose = args.verbose
        self.Ks = eval(args.Ks)


    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def call(self, inputs, **kwargs):
        return

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.random_normal_initializer(stddev=0.01)  # tf.contrib.layers.xavier_initializer()
        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),name='item_embedding')
        print('using random initialization')  # print('using xavier initialization')
        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.add_weight(shape= ([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)
            all_weights['W_bi_%d' % k] = tf.Variable(initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)
            all_weights['W_mlp_%d' % k] = tf.Variable(initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)

        return all_weights

if __name__ == '__main__':
    args = parse_args()
    model = LightGCN(args)
    model.summary()