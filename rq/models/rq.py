"""多层残差向量量化器（Residual Vector Quantizer）。

阅读入口：forward。输入 x=(B,e_dim)，每层对 residual 做一次量化并从 residual
中减掉该层重建；最终 x_q=(B,e_dim)，all_indices=(B,L)，L 是量化层数。

从 ResidualVectorQuantizer.forward 的循环看起：每层用当前 residual 选
code，减去本层量化向量，并累加 x_q；最后 stack 索引得到 (B,L)。
__init__ 中 n_e_list 的长度决定层数 L；每层 VectorQuantizer 有独立码本。
"""

import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 0.25,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100,):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        # n_e_list=[256,256,256] 表示有 3 层量化器，每层各自拥有一个 256×e_dim
        # 的 codebook；所以最终 SID 长度是 3，而不是把 3 层合并成一个码本。
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        # ModuleList 能让 PyTorch 注册每个 codebook 的参数，训练时一起更新。
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        beta=self.beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])

    def get_codebook(self):
        """返回所有层的 codebook，形状通常为 `(L,K,e_dim)`。"""
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        """逐层量化 residual，并把每层的编号拼成一个 SID。

        核心思想：第一层只解释 x 中最容易用一个 code 表示的部分；把第一层
        的近似值减掉后，第二层专门解释剩下的误差；后续层继续处理误差。
        因此最终近似值是 `x_q = q1 + q2 + ... + qL`，而不是只使用最后一层。
        """
        all_losses = []
        all_indices = []

        # x_q 是累计重建向量；residual 是“还没有被前面层解释”的误差。
        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            # 输入 residual=(B,e_dim)，本层返回量化向量 x_res=(B,e_dim)、
            # 标量 loss 和本层索引 indices=(B,)。
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            # 去掉本层已经解释的部分，供下一层继续编码残差。
            residual = residual - x_res
            # 累积前面所有层的量化结果，作为最终 decoder 输入。
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        # 每层 loss 先 stack 成 (L,)，再取平均，保持一个可反传的标量。
        mean_losses = torch.stack(all_losses).mean()
        # all_indices 先是 L 个 (B,) 列表，沿最后一维堆叠后变成 (B,L)。
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices
