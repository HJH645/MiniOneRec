"""RQ-VAE 使用的多层感知机和张量层工具。

阅读入口：MLPLayers。这里负责把输入维度逐层映射到编码器或解码器需要的维度，
不负责数据读取、量化或训练调度；输入输出的最后一维由 layers 参数决定。

先看 MLPLayers.forward：它按 __init__ 里构造的层序列映射最后一维，
用于 RQVAE 的 encoder/decoder。再看 kmeans 和 sinkhorn_algorithm：
前者初始化码本，后者在启用约束分配时被 VectorQuantizer.forward 调用。
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from sklearn.cluster import KMeans


class MLPLayers(nn.Module):
    """按给定维度列表堆叠 Linear、可选 BatchNorm、激活和 Dropout。"""

    def __init__(
        self, layers, dropout=0.0, activation="relu", bn=False
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        # layers=[2560,2048,1024,32] 会创建 3 个 Linear，逐层改变最后一维。
        # 最后一层不加激活，避免限制 latent/重建向量的取值范围。
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))

            if self.use_bn and idx != (len(self.layers)-2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))

            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers)-2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        # 输入通常是 (B, layers[0])，Sequential 只作用于最后一维，输出为
        # (B, layers[-1])；encoder 和 decoder 共用这一套封装。
        return self.mlp_layers(input_feature)

def activation_layer(activation_name="relu", emb_dim=None):

    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
):
    # sklearn 在 CPU/NumPy 上执行 K-means，因此这里会暂时 detach 并搬到 CPU；
    # 返回的聚类中心再放回原 Tensor 的 device。
    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters = num_clusters, max_iter = num_iters).fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)

    return tensor_centers


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    """把距离矩阵转成近似均衡的软分配矩阵 Q。"""
    # distances=(M,K)：M 个样本到 K 个 code 的距离；指数变换后距离越小权重越大。
    Q = torch.exp(- distances / epsilon)

    B = Q.shape[0] # number of samples to assign
    K = Q.shape[1] # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    # print(Q.sum())
    for it in range(sinkhorn_iterations):

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K


    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q
