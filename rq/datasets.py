"""RQ-VAE 的最小 Dataset 封装。

阅读入口：EmbDataset.__init__ 和 __getitem__。np.load 得到 embeddings=(N,D)，
DataLoader 把若干条向量组成 batch=(B,D)；它不负责文本处理，也不负责 SID。

从 EmbDataset.__init__ 看 np.load 如何取得 (N,D)；再看 __getitem__ 如何
按索引返回一条 (D,) 向量；最后看 __len__ 如何告诉 DataLoader 样本数。
这里不读原始文本，输入已经是文本编码器保存的 .npy。
"""

import numpy as np
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):
    """把 `.npy` 商品 embedding 暴露成 PyTorch Dataset。

    `embeddings` 是 `(N,D)`；第 `i` 行对应整数商品 `i`。`__getitem__` 返回
    一条 `(D,)` 向量，DataLoader 再把多条堆成 `(B,D)`。本类不处理文本或 SID。
    """

    def __init__(self, data_path):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        # 输入已经由文本编码器生成；这里仅加载并检查数值是否可用于训练。
        self.embeddings = np.load(data_path)
        
        # Check for NaN values and handle them
        nan_mask = np.isnan(self.embeddings)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN values in embeddings")
            # Replace NaN with zeros
            self.embeddings[nan_mask] = 0.0
            
        # Check for infinite values
        inf_mask = np.isinf(self.embeddings)
        if inf_mask.any():
            print(f"Warning: Found {inf_mask.sum()} infinite values in embeddings")
            # Replace inf with zeros
            self.embeddings[inf_mask] = 0.0
            
        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(f"Embeddings stats - min: {self.embeddings.min():.6f}, max: {self.embeddings.max():.6f}, mean: {self.embeddings.mean():.6f}")
        
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        # 单条样本是 (D,)；DataLoader 批处理后自动堆叠为 (B,D)。
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
