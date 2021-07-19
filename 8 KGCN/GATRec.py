import pandas as pd
from torch.utils.data import DataLoader
import dataloader4graph
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from sklearn.metrics import precision_score,recall_score,accuracy_score


class GAT4Rec(torch.nn.Module):
    def __init__(self, n_users, n_entitys, dim):

        super(GAT4Rec, self).__init__()

        self.entitys = nn.Embedding(n_entitys, dim, max_norm=1)
        self.users = nn.Embedding(n_users, dim, max_norm=1)

        self.multiHeadNumber = 2

        self.W = nn.Linear(in_features=dim, out_features=dim//self.multiHeadNumber, bias=False)
        self.a = nn.Linear(in_features=dim, out_features=1, bias=False)
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)

    def oneHeadAttention(self,target_embeddings, neighbor_entitys_embeddings):
        # t: [batch_size, 1, dim ]
        # n: [batch_size, n_neighbor, dim]
        # [batch_size, w_dim ]
        target_embeddings_w = self.W(target_embeddings)
        # [batch_size, n_neighbor, w_dim]
        neighbor_entitys_embeddings_w = self.W(neighbor_entitys_embeddings)
        # [batch_size, n_neighbor, w_dim]
        target_embeddings_broadcast = torch.cat(
            [torch.unsqueeze(target_embeddings_w, 1) for _ in range(neighbor_entitys_embeddings.shape[1])], dim=1)
        # [batch_size, n_neighbor, w_dim*2]
        cat_embeddings = torch.cat([target_embeddings_broadcast, neighbor_entitys_embeddings_w], 2)
        # [batch_size, n_neighbor, 1]
        eijs = self.leakyRelu(self.a(cat_embeddings))
        # [batch_size, n_neighbor, 1]
        aijs = torch.softmax(eijs, dim=1)
        # [batch_size, w_dim]
        out = torch.sum(aijs * neighbor_entitys_embeddings_w, dim=1)
        return out

    def multiHeadAttentionAggregator(self,target_embeddings, neighbor_entitys_embeddings):
        embs=[]
        for i in range(self.multiHeadNumber):
            embs.append(self.oneHeadAttention(target_embeddings,neighbor_entitys_embeddings))
        return torch.cat(embs,dim=-1)

    def __getEmbedingByNeibourIndex(self,values,neibourIndexs,aggEmbeddings):
        new_values=[]
        for v in values:
            embs=aggEmbeddings[torch.squeeze(torch.LongTensor(neibourIndexs.loc[v].values))]
            new_values.append(torch.unsqueeze(embs,dim=0))
        return torch.cat(new_values,dim=0)

    def gnnForward(self,graph_maps):
        n_hop = 0
        for df in graph_maps:
            if n_hop == 0:
                entity_embs = self.entitys(torch.LongTensor(df.values))
            else:
                entity_embs = self.__getEmbedingByNeibourIndex(df.values, neibourIndexs, aggEmbeddings)
            target_embs = self.entitys(torch.LongTensor(df.index))
            if n_hop < len(graph_maps):
                neibourIndexs = pd.DataFrame(range(len(df.index)), index=df.index)
            aggEmbeddings = self.multiHeadAttentionAggregator(target_embs, entity_embs)
        return aggEmbeddings

    def forward(self,u,graph_maps):
        # [batch_size, dim]
        items = self.gnnForward(graph_maps)
        # [batch_size, dim]
        users = self.users(u)
        # [batch_size]
        uv = torch.sum(users * items, dim=1)
        # [batch_size]
        logit = torch.sigmoid(uv)
        return logit

@torch.no_grad()
def doEva(net,d,G):
    net.eval()
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    i_index = i.detach().numpy()
    maps = dataloader4graph.graphSage4RecAdjType(G,i_index)
    out = net(u,maps)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = r.detach().numpy()
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc

def train(epoch=20,batchSize=1024,dim=128,lr=0.01,eva_per_epochs=1):
    user_set, item_set, train_set, test_set = dataloader4graph.readRecData()
    entitys, pairs = dataloader4graph.readGraphData()
    G = dataloader4graph.get_graph(pairs)
    net = GAT4Rec(max(user_set)+1, max(entitys)+1,dim)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    for e in range(epoch):
        net.train()
        all_lose = 0
        for u, i, r in tqdm(DataLoader(train_set, batch_size=batchSize, shuffle=True)):
            r = torch.FloatTensor(r.detach().numpy())
            optimizer.zero_grad()
            i_index = i.detach().numpy()
            maps = dataloader4graph.graphSage4RecAdjType(G,i_index)
            logits = net(u,maps)
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