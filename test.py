import tensorflow as tf
loss = tf.keras.losses.MeanSquaredError()
class DSSM(tf.keras.Model):
    def __init__(self):
        super(DSSM, self).__init__()
        a = tf.constant([[0.05241987],[0.        ],[0.09177767],[0.        ],[0.        ],[0.        ],[0.26973426],[0.        ],[0.30505422],[0.        ],[0.        ],[0.006693  ],[0.01691625],[0.14264762],[0.        ],[0.15810698]], shape=(16, 1), dtype=float)

        a = tf.reshape(a,shape=(16,))
        # b = self.add_weight(shape=(1, 16),initializer='zero', name='bias')
        b = tf.constant([4, 5, 5, 4, 1, 5, 3, 3, 3, 3, 5, 2, 3, 5, 5, 4], shape=(16,), dtype=float)
        # loss_object = tf.keras.losses.MeanSquaredError()
        # # loss = loss_object([1,2],[3,4])
        # # print(loss)
        # print(tf.sqrt(b))
        # # 点乘
        # # a = tf.multiply(a,b)
        # # 内积
        # # a = tf.reduce_sum(a,axis=1)
        # # print(b)
        # # print(a)
        # # print(loss_object(a,b))
        # # print(tf.sqrt(loss_object(a,b)))
        #
        # c = tf.square(a - b)
        # print(c)
        # c = tf.reduce_sum(c)
        # d = tf.constant([16], dtype=float)
        # print(c)
        # d = tf.divide(c,d)
        # d = tf.sqrt(d)
        # print(d.numpy()[0])
        # print(b)
        # print(a+b)
        print(a.numpy())

if __name__ == '__main__':
    DSSM()