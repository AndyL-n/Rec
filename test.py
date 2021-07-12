import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#
# a = np.array([[0.9,0.035,0.065],[0.15,0.8,0.05],[0.25,0.25,0.5]])
# b = np.array([[1,0,0],[0,1,0],[0,0,1]])
# c= np.array([[1,0,0],[0,1,0],[0,0,1]])
#
# a = 0.85*a + 0.15*c
# print(a)
# for i in range(2):
#     b = np.dot(a,b)
#     print(b,sum(sum(b)))

print(tf.__version__)