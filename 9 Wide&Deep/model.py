"""
author: L
date: 2021/8/18 10:02
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Dropout, Input, Layer
from tensorflow.keras.regularizers import l2

class Linear(Layer):
    def __init__(self, feature_length, w_reg=1e-6):
        """
        Linear Part
        :param feature_length: A scalar. The length of features.
        :param w_reg: A scalar. The regularization coefficient of parameter w.
        """
        super(Linear, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg

    def build(self, input_shape):
        self.w = self.add_weight(name="w",shape=(self.feature_length, 1),regularizer=l2(self.w_reg),trainable=True)

    def call(self, inputs, **kwargs):
        # tf.print(inputs)
        # tf.print(tf.nn.embedding_lookup(self.w, inputs))
        result = tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # tf.print(result)
        return result


class DNN(Layer):
    def __init__(self, hidden_units, activation='relu', dropout=0.):
        """Deep Neural Network
		:param hidden_units: A list. Neural network hidden units.
		:param activation: A string. Activation function of dnn.
		:param dropout: A scalar. Dropout number.
		"""
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x

class WideDeep(Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',dnn_dropout=0., embed_reg=1e-6, w_reg=1e-6):
        """
        Wide&Deep
        :param feature_columns: A list. sparse column feature information.
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param w_reg: A scalar. The regularizer of Linear.
        """
        super(WideDeep, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],input_length=1,
                                         output_dim=feat['embed_dim'],embeddings_initializer='random_uniform',embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)}
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        print(self.index_mapping, self.feature_length)
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.linear = Linear(self.feature_length, w_reg=w_reg)
        self.final_dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](inputs[:, i]) for i in range(inputs.shape[1])], axis=-1)
        x = sparse_embed  # (batch_size, field * embed_dim)
        # Wide
        wide_inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        wide_out = self.linear(wide_inputs)
        # Deep
        deep_out = self.dnn_network(x)
        deep_out = self.final_dense(deep_out)
        # out
        outputs = tf.nn.sigmoid(0.5 * wide_out + 0.5 * deep_out)
        return outputs

    def summary(self, **kwargs):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
