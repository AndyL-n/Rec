import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        Factorization Machines
        :param feature_columns: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        """
        super(FM_Layer, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.feature_length, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        # mapping
        inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        # first order
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # second order
        second_inputs = tf.nn.embedding_lookup(self.V, inputs)  # (batch_size, fields, embed_dim)
        square_sum = tf.square(tf.reduce_sum(second_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(second_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        # outputs
        outputs = first_order + second_order
        return outputs


class FM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        Factorization Machines
        :param feature_columns: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
		:param v_reg: the regularization coefficient of parameter v
        """
        super(FM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.fm = FM_Layer(feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        fm_outputs = self.fm(inputs)
        outputs = tf.nn.sigmoid(fm_outputs)
        return outputs

    def summary(self, **kwargs):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()

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
    def dataset(self, file, embed_dim=8, read_part=True, sample_num=100000, test_size=0.2):
        """
        a example about creating criteo dataset
        :param file: dataset's path
        :param embed_dim: the embedding dimension of sparse features
        :param read_part: whether to read part of it
        :param sample_num: the number of instances if read_part is True
        :param test_size: ratio of test dataset
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
        train, test = train_test_split(data_df, test_size=test_size)

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
    file = '../dataset/Criteo/train.txt'
    read_part = True
    sample_num = 5000000
    test_size = 0.2

    k = 8

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10
    # ========================== Create dataset =======================
    feature_columns, train, test = loading.dataset(file=file,
                                           read_part=read_part,
                                           sample_num=sample_num,
                                           test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    model = FM(feature_columns=feature_columns, k=k)
    model.summary()
    # ============================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                      metrics=[AUC()])
    # ============================model checkpoint======================
    # check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
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