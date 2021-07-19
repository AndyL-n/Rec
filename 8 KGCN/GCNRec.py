from torch.utils.data import DataLoader
import dataloader4graph
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.metrics import precision_score,recall_score,accuracy_score

class GCN4Rec(torch.nn.Module):
    def __init__(self, n_users, n_entitys, dim, hidden_dim):
        '''
        :param n_classes: 类别数
        :param dim: 特征维度
        '''
        super(GCN4Rec, self).__init__()

        # 随机初始化所有节点向量(KG图 包括物品向量)
        self.entitys = nn.Embedding(n_entitys, dim, max_norm=1)

        # 随机初始化所有用户向量
        self.users = nn.Embedding(n_users, dim, max_norm=1)

        # 所有节点的索引
        self.all_entitys_indexes = torch.LongTensor(range(n_entitys))

        # 初始化两个 GCN 层
        self.conv1 = GCNConv(dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, dim)

    def gnnForward(self,i,edges):
        '''
        'param i: 物品索引[batch_size]
        'param edges: 表示图的边集
        '''
        # [n_entitys, dim]
        x = self.entitys(self.all_entitys_indexes)
        # [n_entitys, hidden_dim]
        x = self.conv1(x, edges)
        x = F.relu(x)
        x = F.dropout(x)
        # [n_entitys, dim]
        x = self.conv2(x, edges)
        # 通过物品索引取出
        # [batch_size, dim]
        return x[i]

    def forward(self,u,i,edges):
        # [batch_size * dim]
        items = self.gnnForward(i,edges)
        # [batch_size * dim]
        users = self.users(u)
        # [batch_size * 1]
        uv = torch.sum(users * items, dim=1)
        logit = torch.sigmoid(uv)
        return logit


@torch.no_grad()
def doEva(net,d,G):
    net.eval()
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    i_index = i.detach().numpy()
    edges = dataloader4graph.graphSage4Rec(G, i_index)
    out = net(u,i,edges)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = r.detach().numpy()
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc

def train(epoch=20,batchSize=1024,dim=128,hidden_dim=64,lr=0.01,eva_per_epochs=1):
    user_set, item_set, train_set, test_set = dataloader4graph.readRecData()
    entitys, pairs = dataloader4graph.readGraphData()
    G = dataloader4graph.get_graph(pairs)
    net = GCN4Rec(max(user_set)+1, max(entitys)+1,dim,hidden_dim)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    for e in range(epoch):
        net.train()
        all_lose = 0
        for u, i, r in tqdm(DataLoader(train_set, batch_size=batchSize, shuffle=True)):
            r = torch.FloatTensor(r.detach().numpy())
            optimizer.zero_grad()
            i_index = i.detach().numpy()
            edges = dataloader4graph.graphSage4Rec(G,i_index)
            logits = net(u,i,edges)
            loss = criterion(logits,r)
            all_lose+=loss
            loss.backward()
            optimizer.step()

        print('epoch {}, avg_loss = {:.4f}'.format(e, all_lose / (len(train_set) // batchSize)))

        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(net, train_set,G)
            print('train: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))
            p, r, acc = doEva(net, test_set,G)
            print('test: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))


    return net



if __name__ == '__main__':
    train()