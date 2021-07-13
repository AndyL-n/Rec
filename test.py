import tensorflow as tf
loss = tf.keras.losses.MeanSquaredError()
class DSSM(tf.keras.Model):
    def __init__(self):
        super(DSSM, self).__init__()
        a = tf.constant([[2,3],[4,5]], dtype=float)
        # b = self.add_weight(shape=(1, 16),initializer='zero', name='bias')
        b = tf.constant([[ 1,1],[1,1]], dtype=float)
        # 点乘
        a = tf.multiply(a,b)
        # 内积
        a = tf.reduce_sum(a,axis=1)
        print(a)
        # print(b)
        # print(a+b)

if __name__ == '__main__':
    DSSM()