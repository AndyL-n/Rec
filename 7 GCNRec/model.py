"""
author: L
date: 2021/8/19 14:48
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dropout, Layer, Input

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class model(Model):
    def __init__(self):
        super(model, self).__init__()


    def call(self, inputs, **kwargs):
        print(inputs)
        return inputs

    def summary(self, **kwargs):
        inputs = Input(shape=(1,1,1), dtype=tf.int32, name="w")
        Model(inputs=inputs, outputs=self.call(inputs)).summary()

def parse_args():
    parser = argparse.ArgumentParser(description="params")
    return parser

if __name__ == '__main__':
    args = parse_args()
    model = model()
    model.summary()
    model([1,2])
