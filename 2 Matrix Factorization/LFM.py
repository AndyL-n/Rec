import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
import numpy as np
import os
import sys
import warnings
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MF_layer(Layer):
    def __init__(self, user_num, item_num, latent_dim, use_bias=False, user_reg=1e-4, item_reg=1e-4,
                 user_bias_reg=1e-4, item_bias_reg=1e-4):
        """
        MF Layer
        :param user_num: user length
        :param item_num: item length
        :param latent_dim: latent number
        :param use_bias: whether using bias or not
        :param user_reg: regularization of user
        :param item_reg: regularization of item
        :param user_bias_reg: regularization of user bias
        :param item_bias_reg: regularization of item bias
        """
        super(MF_layer, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.use_bias = use_bias
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.user_bias_reg = user_bias_reg
        self.item_bias_reg = item_bias_reg

    def build(self, input_shape):
        self.p = self.add_weight(name='user_latent_matrix',
                                 shape=(self.user_num, self.latent_dim),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.user_reg),
                                 trainable=True)
        self.q = self.add_weight(name='item_latent_matrix',
                                 shape=(self.item_num, self.latent_dim),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.item_reg),
                                 trainable=True)
        self.user_bias = self.add_weight(name='user_bias',
                                         shape=(self.user_num, 1),
                                         initializer=tf.random_normal_initializer(),
                                         regularizer=l2(self.user_bias_reg),
                                         trainable=self.use_bias)
        self.item_bias = self.add_weight(name='item_bias',
                                         shape=(self.item_num, 1),
                                         initializer=tf.random_normal_initializer(),
                                         regularizer=l2(self.item_bias_reg),
                                         trainable=self.use_bias)

    def call(self, inputs, **kwargs):
        user_id, item_id, avg_score = inputs
        # MF
        latent_user = tf.nn.embedding_lookup(params=self.p, ids=user_id)
        latent_item = tf.nn.embedding_lookup(params=self.q, ids=item_id)
        outputs = tf.reduce_sum(tf.multiply(latent_user, latent_item), axis=1, keepdims=True)
        # MF-bias
        user_bias = tf.nn.embedding_lookup(params=self.user_bias, ids=user_id)
        item_bias = tf.nn.embedding_lookup(params=self.item_bias, ids=item_id)
        bias = tf.reshape((avg_score + user_bias + item_bias), shape=(-1, 1))
        # use bias
        outputs = bias + outputs if self.use_bias else outputs
        return outputs

    def summary(self):
        user_id = tf.keras.Input(shape=(), dtype=tf.int32)
        item_id = tf.keras.Input(shape=(), dtype=tf.int32)
        avg_score = tf.keras.Input(shape=(), dtype=tf.float32)
        tf.keras.Model(inputs=[user_id, item_id, avg_score], outputs=self.call([user_id, item_id, avg_score])).summary()

class MF(tf.keras.Model):
    def __init__(self, feature_columns,
                 implicit=False,
                 use_bias=False,
                 user_reg=1e-4,
                 item_reg=1e-4,
                 user_bias_reg=1e-4,
                 item_bias_reg=1e-4):
        """
        MF Model
        :param feature_columns: dense_feature_columns + sparse_feature_columns
        :param implicit: whether implicit or not
        :param use_bias: whether using bias or not
        :param user_reg: regularization of user
        :param item_reg: regularization of item
        :param user_bias_reg: regularization of user bias
        :param item_bias_reg: regularization of item bias
        """
        super(MF, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        num_users = self.sparse_feature_columns[0]['feat_num']
        num_items = self.sparse_feature_columns[1]['feat_num']
        latent_dim = self.sparse_feature_columns[0]['embed_dim']
        self.mf_layer = MF_layer(user_num = num_users,
                                 item_num = num_items,
                                 latent_dim = latent_dim,
                                 use_bias = use_bias,
                                 user_reg = user_reg,
                                 item_reg = item_reg,
                                 user_bias_reg = user_bias_reg,
                                 item_bias_reg = item_bias_reg)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        user_id, item_id = sparse_inputs[:, 0], sparse_inputs[:, 1]
        avg_score = dense_inputs
        outputs = self.mf_layer([user_id, item_id, avg_score])
        return outputs

    def summary(self, **kwargs):
        dense_inputs = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()

class loading():
    def sparseFeature(self, feat, feat_num, embed_dim=4):
        """
        create dictionary for sparse feature
        :param feat: feature name
        :param feat_num: the total number of sparse features that do not repeat
        :param embed_dim: embedding dimension
        :return:
        """
        return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

    def denseFeature(self, feat):
        """
        create dictionary for dense feature
        :param feat: dense feature name
        :return:
        """
        return {'feat': feat}

    def dataset(self, file, latent_dim = 4, pivot = 0.2):
        """
        create the explicit dataset of movielens-1m
        We took the last 20% of each user sorted by timestamp as the test dataset
        Each of these samples contains UserId, MovieId, Rating, avg_score
        :param file: dataset path
        :param latent_dim: latent factor
        :param test_size: ratio of test dataset
        :return: user_num, item_num, train_df, test_df
        """
        data_df = pd.read_csv(file, sep="::", engine='python',
                              names=['UserId', 'MovieId', 'Rating', 'Timestamp'])
        data_df['avg_score'] = data_df.groupby(by='UserId')['Rating'].transform('mean')
        # feature columns
        user_num, item_num = data_df['UserId'].max() + 1, data_df['MovieId'].max() + 1
        feature_columns = [[self.denseFeature('avg_score')],
                           [self.sparseFeature('user_id', user_num, latent_dim),
                            self.sparseFeature('item_id', item_num, latent_dim)]]
        # split train dataset and test dataset
        watch_count = data_df.groupby(by='UserId')['MovieId'].agg('count')
        print("分割后"+str(pivot*100)+"%作为数据集\n")
        test_df = pd.concat([data_df[data_df.UserId == i].iloc[int((1 - pivot) * watch_count[i]):] for i in tqdm(watch_count.index)], axis=0)
        print(test_df.head())
        test_df = test_df.reset_index()
        train_df = data_df.drop(labels=test_df['index'])
        # 删除非需求列
        train_df = train_df.drop(['Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)
        test_df = test_df.drop(['index', 'Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)
        train_X = [train_df['avg_score'].values, train_df[['UserId', 'MovieId']].values]
        train_y = train_df['Rating'].values.astype('int32')
        test_X = [test_df['avg_score'].values, test_df[['UserId', 'MovieId']].values]
        test_y = test_df['Rating'].values.astype('int32')
        return feature_columns, (train_X, train_y), (test_X, test_y)

if __name__ == '__main__':
    # Parameters
    rating_file = '../data/ml-1m/ratings.dat'
    pivot = 0.2
    latent_dim = 32
    use_bias = True
    learning_rate = 0.001
    batch_size = 512
    epochs = 10

    # Dataset
    load = loading()
    feature_columns, train, test = load.dataset(rating_file, latent_dim, pivot)
    # print(feature_columns)
    train_X, train_y = train
    test_X, test_y = test
    # Model
    model = MF(feature_columns, use_bias)
    model.summary()

    # Checkpoint
    # check_path = '../save/mf_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only = True, verbose=1, period=5)

    # Compile
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate),metrics=['mse'])
    # Train
    model.fit(train_X,train_y,
        epochs=epochs,
        # callbacks=[checkpoint],
        batch_size=batch_size,
        validation_split=0.1
    )
    # Test
    print('test rmse: %f' % np.sqrt(model.evaluate(test_X, test_y)[1]))