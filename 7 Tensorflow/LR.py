# y=2x

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
import matplotlib.pyplot as plt

train_X =np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
# plt.plot(train_X, train_Y, 'ro', label='data')
# plt.legend()
# plt.show()
dataset = tf.data.Dataset.from_tensor_slices((np.reshape(train_X, [-1,1]), np.reshape(train_Y, [-1,1])))

print(dataset)
