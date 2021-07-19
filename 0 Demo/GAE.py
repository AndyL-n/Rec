import tensorflow as tf
import sys

class NeighborAggregator(tf.keras.Model):
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_method="mean"):
        """聚合节点邻居
        Args:
            input_dim: 输入特征的维度
            output_dim: 输出特征的维度
            use_bias: 是否使用偏置 (default: {False})
            aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(NeighborAggregator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method

        self.weight = self.add_weight(shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform', name='kernel')

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.input_dim, self.output_dim),
                                        initializer='zero', name='bias')

    def call(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = tf.math.reduce_mean(neighbor_feature, axis=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = tf.math.reduce_sum(neighbor_feature, axis=1)
        elif self.aggr_method == "max":
            aggr_neighbor = tf.math.reduce_max(neighbor_feature, axis=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}".format(self.aggr_method))
        neighbor_hidden = tf.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

class SageGCN(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, activation=tf.keras.activations.relu,
                 aggr_neighbor_method="mean", aggr_hidden_method="sum"):
        """SageGCN层定义
        Args:
            input_dim: 输入特征的维度
            hidden_dim: 隐层特征的维度，
                当aggr_hidden_method=sum, 输出维度为hidden_dim
                当aggr_hidden_method=concat, 输出维度为hidden_dim*2
            activation: 激活函数
            aggr_neighbor_method: 邻居特征聚合方法，["mean", "sum", "max"]
            aggr_hidden_method: 节点特征的更新方法，["sum", "concat"]
        """
        super(SageGCN, self).__init__()

        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim,
                                             aggr_method=aggr_neighbor_method)

        self.weight = self.add_weight(shape=(self.input_dim, self.hidden_dim),
                                      initializer='glorot_uniform',
                                      name='kernel')

    def call(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = tf.matmul(src_node_features, self.weight)

        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = tf.concat(1, [self_hidden, neighbor_hidden])
        else:
            raise ValueError("Expected sum or concat, got {}".format(self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

class GraphSage(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim,num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.gcn = []
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index + 1]))
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))
    def call(self, node_features_list):
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = tf.reshape(hidden[hop + 1], (src_node_num, self.num_neighbors_list[hop], -1))
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]


class DSSM(tf.keras.Model):
    def __init__(self, input_dim_user, input_dim_item, hidden_dim, num_neighbors_list):
        super(DSSM, self).__init__()
        self.input_dim_user = input_dim_user
        self.input_dim_item = input_dim_item
        # 隐藏单元节点数
        self.hidden_dim = hidden_dim
        # 每阶采样邻居的节点数
        self.num_neighbors_list = num_neighbors_list
        # 层数
        self.num_layers = len(self.num_neighbors_list)
        self.GAE_user = GraphSage(input_dim=self.input_dim_user, hidden_dim=self.hidden_dim, num_neighbors_list=self.num_neighbors_list)
        self.GAE_item = GraphSage(input_dim=self.input_dim_item, hidden_dim=self.hidden_dim, num_neighbors_list=self.num_neighbors_list)
        # MLP
        self.w1 = self.add_weight(shape=(self.hidden_dim[-1] * 2, self.hidden_dim[-1]),
                                      initializer='glorot_uniform', name='mlp_weight1')
        self.b1 = self.add_weight(shape=(1, self.hidden_dim[-1]),
                                        initializer='zero', name='mlp_bias1')
        self.w2 = self.add_weight(shape=(self.hidden_dim[-1], 8),
                                      initializer='glorot_uniform', name='mlp_weight2')
        self.b2 = self.add_weight(shape=(1, 8),
                                        initializer='zero', name='mlp_bias2')
        self.w3 = self.add_weight(shape=(8, 1),
                                      initializer='glorot_uniform', name='mlp_weight3')
        self.b3 = self.add_weight(shape=(1, 1),
                                        initializer='zero', name='mlp_bias3')
    def call(self, user_node_features_list, item_node_features_list):
        user = self.GAE_user(user_node_features_list)
        item = self.GAE_item(item_node_features_list)
        # print("------------------user&item-------------------------")
        # print(user)
        # print(user.dtype)
        # print(item)
        # print(item.dtype)
        # print("-------------------tmp------------------------------")
        tmp = tf.concat([user,item], axis=1)
        # print(tmp)
        # print(tmp.dtype)
        # print("-------------------MLP------------------------------")
        # print(tf.matmul(tmp, self.w1))
        # print(self.b1)
        MLP1 = tf.matmul(tmp, self.w1) + self.b1
        # print(MLP1)
        # print(MLP1.dtype)
        # print("----------------------------------------------------")
        MLP2 = tf.matmul(MLP1, self.w2) + self.b2
        MLP3 = tf.matmul(MLP2, self.w3) + self.b3
        # print("----------------------------------------------------")
        # print(MLP3)
        # print("----------------------------------------------------")
        output = tf.keras.activations.relu(MLP3)
        return output
