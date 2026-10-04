"""Residual Quantized VAE 模型。

阅读入口：forward、get_indices、compute_loss。输入 x=(B,in_dim)，encoder
变为 (B,e_dim)，残差量化器返回 x_q=(B,e_dim)、标量 rq_loss、indices=(B,L)，
decoder 还原 out=(B,in_dim)。get_indices 只返回索引，供生成 index.json。

从 RQVAE.forward 看 encoder → rq → decoder 的 (B,in_dim) → (B,e_dim)
→ (B,in_dim)；再看 compute_loss 如何组合重建误差和量化误差。
get_indices 只走 encoder 与 rq，供训练后生成商品 SID。
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    """把连续商品向量压缩成离散 SID，并支持从 SID 向量重建原向量。

    可以把本类看成三段流水线：
    `encoder` 把高维 embedding 压到较小的连续空间，`rq` 把连续向量替换成
    多个 codebook 向量的和，`decoder` 再尝试还原原始 embedding。训练时希望
    重建结果接近输入，同时让量化向量和 encoder 输出不要相差太远。
    """
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        # 例如 in_dim=2560、layers=[2048,...,64]、e_dim=32 时，encoder 的
        # 维度链是 2560→2048→...→64→32；decoder 使用反向链 32→64→...→2560。
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, use_sk=True):
        """训练或推理一批商品 embedding。

        `x` 是文本编码器输出的连续向量，形状 `(B,in_dim)`。`indices` 的
        每一行是一个商品的离散 SID 片段编号，形状 `(B,L)`，L 等于量化层数。
        """
        # x=(B,in_dim)；z=(B,e_dim) 是 encoder 的连续隐向量。
        x = self.encoder(x)
        # x_q 是多个 codebook 向量之和，rq_loss 是量化器损失，indices 是 SID 编号。
        x_q, rq_loss, indices = self.rq(x,use_sk=use_sk)
        # decoder 只接收量化后的向量，不能直接看到原始 x。
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        """只计算 SID，不做 decoder 和反向传播，供生成 index.json 使用。"""
        # 推理路径与 forward 的前两步相同；@torch.no_grad() 防止保存计算图。
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):
        """返回总损失和重建损失。

        总损失 = 重建损失 + quant_loss_weight × 量化损失。重建损失让 decoder
        保留商品 embedding 信息，量化损失让 codebook 学会贴近 encoder 表示。
        """

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon
