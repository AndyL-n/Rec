import tensorflow as tf
import time
import random
import sys
import numpy as np
from GAE import DSSM
from data_load import CF
from sampling import multihop_sampling
from collections import namedtuple

import matplotlib.pyplot as plt

# 输入维度
INPUT_DIM_USER = 879
INPUT_DIM_ITEM = 91
# Note: 采样的邻居阶数需要与GCN的层数保持一致
HIDDEN_DIM = [128, 16]   # 隐藏单元节点数
NUM_NEIGHBORS_LIST = [20, 10]   # 每阶采样邻居的节点数
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)
BTACH_SIZE = 16     # 批处理大小
EPOCHS = 20
NUM_BATCH_PER_EPOCH = 20    # 每个epoch循环的批次数
LEARNING_RATE = 0.01    # 学习率

user_adj, item_adj,train_set,test_set,user_emb,item_emb = CF().load()
user_emb, item_emb = np.array(user_emb), np.array(item_emb)
model = DSSM(input_dim_user=INPUT_DIM_USER, input_dim_item=INPUT_DIM_ITEM, hidden_dim=HIDDEN_DIM, num_neighbors_list=NUM_NEIGHBORS_LIST)
optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, decay=5e-4)
loss_object = tf.keras.losses.MeanSquaredError()

# 记录过程值，以便最后可视化
train_loss = []
train_val_MSE = []
train_test_MSE = []

def train(t1):
    print("time:{}".format(time.time() - t1))
    for e in range(EPOCHS):
        for batch in range(NUM_BATCH_PER_EPOCH):
            batch_src = random.sample(train_set, BTACH_SIZE)
            batch_src_users = [i[0] for i in batch_src]
            batch_src_items = [i[1] for i in batch_src]
            batch_src_scores = [i[2] for i in batch_src]
            print(batch_src_users, batch_src_items, batch_src_scores)
            batch_sampling_result_user = multihop_sampling(batch_src_users, NUM_NEIGHBORS_LIST, user_adj)
            batch_sampling_result_item = multihop_sampling(batch_src_items, NUM_NEIGHBORS_LIST, item_adj)
            batch_sampline_embedding_user = []
            for id,users in enumerate(batch_sampling_result_user):
                batch_sampline_embedding_user.append([])
                for user in users:
                    batch_sampline_embedding_user[id].append(user_emb[user])
            batch_sampline_embedding_item = []
            for id, items in enumerate(batch_sampling_result_item):
                batch_sampline_embedding_item.append([])
                for item in items:
                    batch_sampline_embedding_item[id].append(item_emb[item])
            # print(batch_sampline_embedding_user)
            # print(batch_sampline_embedding_item)
            loss = 0.0
            with tf.GradientTape() as tape:
                batch_train_scores = model(batch_sampline_embedding_user, batch_sampline_embedding_item)
                loss = loss_object(batch_src_scores, batch_train_scores)
                print("Epoch {:03d} Batch {:03d} loss(MSE):{}".format(e, batch, loss))
                grads = tape.gradient(loss, model.trainable_variables)

                optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_val = test(train_set)
        train_test = test(test_set)
        print("Epoch {:03d} loss: {} val MSE: {} test MSE:{} time:{}".format(e, loss, train_val, train_test, time.time()-t1))

        train_loss.append(loss)
        train_val_MSE.append(train_val)
        train_test_MSE.append(train_test)
        # 训练过程可视化
        fig, axes = plt.subplots(3, sharex=True, figsize=(12, 8))
        fig.suptitle('Training Metrics')
        axes[0].set_ylabel("Loss", fontsize=14)
        axes[0].plot(train_loss)

        axes[1].set_ylabel("Val MSE", fontsize=14)
        axes[1].plot(train_val_MSE)

        axes[2].set_ylabel("Test MSE", fontsize=14)
        axes[2].plot(train_test_MSE)

        plt.show()

def test(Set):
    Set = random.sample(Set, BTACH_SIZE * 4)
    users = [i[0] for i in Set]
    items = [i[1] for i in Set]
    scores = [i[2] for i in Set]
    batch_sampling_result_user = multihop_sampling(users, NUM_NEIGHBORS_LIST, user_adj)
    batch_sampling_result_item = multihop_sampling(items, NUM_NEIGHBORS_LIST, item_adj)
    batch_sampline_embedding_user = []
    for id, users in enumerate(batch_sampling_result_user):
        batch_sampline_embedding_user.append([])
        for user in users:
            batch_sampline_embedding_user[id].append(user_emb[user])
    batch_sampline_embedding_item = []
    for id, items in enumerate(batch_sampling_result_item):
        batch_sampline_embedding_item.append([])
        for item in items:
            batch_sampline_embedding_item[id].append(item_emb[item])
    batch_train_scores = model(batch_sampline_embedding_user, batch_sampline_embedding_item)
    MSE = loss_object(batch_train_scores, scores)
    return MSE
if __name__ == '__main__':
    t1 = time.time()
    train(t1)
    print("time:{}".format(time.time() - t1))

