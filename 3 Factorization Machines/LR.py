import os
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

class LR_layer(Layer):
    def __init__(self, feature_length):
        super(LR_layer,self).__init__()
        self.feature_length = feature_length
        print(self.feature_length)
        print("LR_layer")

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,1),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True)
    def call(self, inputs):
        outputs = self.w0 + tf.reduce_sum(tf.matmul(inputs, self.w), axis=1)
        outputs = tf.sigmoid(outputs)
        return outputs


class LR(tf.keras.Model):
    def __init__(self, feature_length):
        super(LR, self).__init__()
        self.feature_length = feature_length
        self.lr_layer = LR_layer(self.feature_length)
        print("LR")

    def call(self, inputs, **kwargs):
        outputs = self.lr_layer(inputs)
        return outputs

    def summary(self, **kwargs):
        features = tf.keras.Input(shape=(self.feature_length,), dtype=tf.float32)
        tf.keras.Model(inputs=[features], outputs=self.call([features])).summary()

class loading():
    def __init__(self, path, pivot = 0.2):
        self.data_path = os.path.join(path, 'u.data')
        self.user_path = os.path.join(path, 'u.user')
        self.item_path = os.path.join(path, 'u.item')
        self.occupation_path = os.path.join(path, 'u.occupation')
        self.pivot = pivot
        self.user_dict = {}
        self.item_dict = {}
        self.occupation_dict = {}
        self.gender_dict = {'M': 1, 'F': 0}

    # rating 0-1映射
    def getCtr(self, rating):
        return 1.0 if rating > 3 else 0.0

    def read_rating_data(self):
        dataSet = {}
        with open(self.data_path, 'r') as f:
            for line in f.readlines():
                d = line.strip().split('\t')
                dataSet[(int(d[0]), int(d[1]))] = [self.getCtr(int(d[2]))]
        return dataSet

    def read_item_hot(self):
        items = {}
        with open(self.item_path, 'r', encoding='ISO-8859-1') as f:
            for line in f.readlines():
                d = line.strip().split('|')
                items[int(d[0])] = np.array(d[5:], dtype='float64')
        return items

    def read_occupation_hot(self):
        occupations = {}
        with open(self.occupation_path, 'r') as f:
            names = f.read().strip().split('\n')
        length = len(names)
        # genre one_hot
        for i in range(length):
            l = np.zeros(length, dtype='float64')
            l[i] = 1
            occupations[names[i]] = l
        return occupations

    def read_user_hot(self):
        users = {}
        self.occupation_dict = self.read_occupation_hot()
        with open(self.user_path, 'r') as f:
            for line in f.readlines():
                d = line.strip().split('|')
                a = np.array([int(d[1]), self.gender_dict[d[2]]])
                users[int(d[0])] = np.append(a, self.occupation_dict[d[3]])
        return users

    def read_dataSet(self):
        trainX, trainY, testX, testY = [], [], [], []
        ratings = self.read_rating_data()
        for index,k in enumerate(ratings):
            if ((index % 10) / 10 < self.pivot):
                testX.append(np.append(self.user_dict[k[0]], self.item_dict[k[1]]))
                testY.append(ratings[k])
            else:
                trainX.append(np.append(self.user_dict[k[0]], self.item_dict[k[1]]))
                trainY.append(ratings[k])
        return trainX, trainY, testX, testY

    def dataset(self):
        # item one_hot embedding
        self.item_dict = self.read_item_hot()
        # user one_hot embedding
        self.user_dict = self.read_user_hot()
        return self.read_dataSet()

if __name__ == '__main__':
    file_path = '../data/ml-100k'
    pivot = 0.2
    load = loading(file_path, pivot)
    trainX, trainY, testX, testY=load.dataset()
    print(trainX[0], trainY[0])
    feature_length = len(trainX[0])
    batch_size= 512
    learning_rate = 0.001
    epochs = 10
    model = LR(feature_length)

    model.summary()
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
    model.fit(np.array(trainX), np.array(trainY),
        epochs=epochs,
        # callbacks=[checkpoint],
        batch_size=batch_size,
        validation_split=0.1
    )

