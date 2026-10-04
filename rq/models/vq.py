"""单层 VectorQuantizer：把连续向量分配到有限 codebook。

阅读入口：forward。输入 x=(B,...,e_dim) 会 flatten 成 latent=(-1,e_dim)，
距离矩阵是 (-1,n_e)，indices 是 (...,)，量化向量恢复为与 x 同形状，loss 是标量。

从 VectorQuantizer.forward 看距离矩阵 d=(样本位置数,码本大小)；
argmin 或 Sinkhorn 选索引；embedding(indices) 取量化向量；再看
commitment/codebook 两项损失和直通梯度。init_emb 只在启用 kmeans_init
且训练时初始化码本，get_codebook_entry 用于按索引反查向量。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):
    """单层离散化器：从 codebook 中为每个连续向量选择最近的 code。"""

    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.003, sk_iters=100,):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        # embedding.weight 就是 codebook，形状 (n_e,e_dim)；第 j 行是 code j 的向量。
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        """返回本层全部 code 向量，形状 `(n_e,e_dim)`。"""
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # indices 是 code 编号；Embedding 查表后得到对应的量化向量。
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):
        """用当前 batch 的 latent 做 K-means，把聚类中心初始化为 codebook。"""
        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        """把连续向量映射为 codebook 向量，并返回索引和量化损失。

        对每个 latent 计算它到所有 codebook 行的平方欧氏距离，通常取距离最小
        的 code。`use_sk=True` 且 sk_epsilon>0 时改用 Sinkhorn 做近似均衡分配。
        """
        # x=(B,e_dim) 时 latent=(B,e_dim)；更高维输入会展平前面的维度，保留最后
        # 一维作为向量维度，最后再把索引恢复成 x.shape[:-1]。
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # 利用 ||a-b||²=||a||²+||b||²-2a·b 一次性得到距离矩阵 d=(M,n_e)，
        # M 是展平后的向量数，n_e 是本层 code 数量。
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.weight.t())
        if not use_sk or self.sk_epsilon <= 0:
            # 普通最近邻量化：每个 latent 独立选择距离最小的 code。
            indices = torch.argmin(d, dim=-1)
        else:
            # Sinkhorn 路径把距离变成近似均衡的 assignment，减少所有商品挤到
            # 少数 code 的情况；Q 的形状仍为 (M,n_e)，argmax 得到 code 编号。
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        # indices = torch.argmin(d, dim=-1)

        # 查 codebook 得到量化向量，并恢复成与输入 x 完全相同的形状。
        x_q = self.embedding(indices).view(x.shape)

        # codebook_loss 更新 codebook 使其靠近输入；commitment_loss 约束 encoder
        # 输出靠近选中的 code。detach 让两项损失分别作用在预期的参数上。
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # 直通估计器：前向使用离散的 x_q，反向把梯度近似传给原始 x，
        # 从而让 encoder 可以通过不可导的 argmin 继续训练。
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices
