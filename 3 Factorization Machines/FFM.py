import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
import pandas as pd
import os

class FFM_Layer(Layer):
    def __init__(self, sparse_feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        :param dense_feature_columns: A list. sparse column feature information.
        :param k: A scalar. The latent vector
        :param w_reg: A scalar. The regularization coefficient of parameter w
		:param v_reg: A scalar. The regularization coefficient of parameter v
        """
        super(FFM_Layer, self).__init__()
        self.sparse_feature_columns = sparse_feature_columns
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.field_num = len(self.sparse_feature_columns)

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer='random_normal',
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.v = self.add_weight(name='v',
                                 shape=(self.feature_length, self.field_num, self.k),
                                 initializer='random_normal',
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        # first order
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # field second order
        second_order = 0
        latent_vector = tf.reduce_sum(tf.nn.embedding_lookup(self.v, inputs), axis=1)  # (batch_size, field_num, k)
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
                second_order += tf.reduce_sum(latent_vector[:, i] * latent_vector[:, j], axis=1, keepdims=True)
        return first_order + second_order

class FFM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        FFM architecture
        :param feature_columns: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
		:param field_reg_reg: the regularization coefficient of parameter v
        """
        super(FFM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.ffm = FFM_Layer(self.sparse_feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        ffm_out = self.ffm(inputs)
        outputs = tf.nn.sigmoid(ffm_out)
        return outputs

    def summary(self, **kwargs):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()

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
    def dataset(self, file, embed_dim=8, read_part=True, sample_num=100000, pivot=0.2):
        """
        a example about creating criteo dataset
        :param file: dataset's path
        :param embed_dim: the embedding dimension of sparse features
        :param read_part: whether to read part of it
        :param sample_num: the number of instances if read_part is True
        :param pivot: ratio of test dataset
        :return: feature columns, train, test
        """
        names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
                 'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
                 'C23', 'C24', 'C25', 'C26']

        if read_part:
            data_df = pd.read_csv(file, sep='\t', iterator=True, header=None,
                                  names=names)
            data_df = data_df.get_chunk(sample_num)

        else:
            data_df = pd.read_csv(file, sep='\t', header=None, names=names)

        sparse_features = ['C' + str(i) for i in range(1, 27)]
        dense_features = ['I' + str(i) for i in range(1, 14)]
        features = sparse_features + dense_features

        data_df[sparse_features] = data_df[sparse_features].fillna('-1')
        data_df[dense_features] = data_df[dense_features].fillna(0)

        # Bin continuous data into intervals.
        est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
        data_df[dense_features] = est.fit_transform(data_df[dense_features])

        for feat in sparse_features:
            le = LabelEncoder()
            data_df[feat] = le.fit_transform(data_df[feat])

        # ==============Feature Engineering===================

        # ====================================================
        feature_columns = [self.sparseFeature(feat, int(data_df[feat].max()) + 1, embed_dim=embed_dim)
                           for feat in features]
        train, test = train_test_split(data_df, test_size=pivot)
        train_X = train[features].values.astype('int32')
        train_y = train['label'].values.astype('int32')
        test_X = test[features].values.astype('int32')
        test_y = test['label'].values.astype('int32')

        return feature_columns, (train_X, train_y), (test_X, test_y)

if __name__ == '__main__':
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    # If you have GPU, and the value is GPU serial number.
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    # ========================= Hyper Parameters =======================
    # you can modify your file path
    file = '../data/Criteo/train.txt'
    read_part = True
    sample_num = 10000
    pivot = 0.2
    load = loading()
    k = 10

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10

    # ========================== Create dataset =======================
    feature_columns, train, test = load.dataset(file=file,
                                           read_part=read_part,
                                           sample_num=sample_num,
                                           pivot=pivot)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = FFM(feature_columns=feature_columns, k=k)
    model.summary()
    # ============================model checkpoint======================
    # check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ============================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ==============================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])
