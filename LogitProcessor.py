"""约束解码的 logits 处理器。

阅读入口：ConstrainedLogitsProcessor.__call__。无 beam 时 input_ids=(B,L)、
scores=(B,V)；有 K 个 beam 时 Transformers 展平为 (B*K,L)/(B*K,V)。处理器
根据已生成前缀查允许 token，合法位置加 0、非法位置保留 -inf，返回同形状
(B*K,V) 的分数。count 表示当前生成步，每次 generate 前应创建新实例。

从 ConstrainedLogitsProcessor.__call__ 的双层循环看：每个 batch 的每个
beam 取已生成后缀，调用 prefix_allowed_tokens_fn 查合法 token，
用 -inf mask 屏蔽其他词。查不到前缀时仅放行 EOS（若配置了 EOS），
同时发出警告。count 记录本次 generate 的步数，不能跨请求复用实例。
"""

from transformers.generation import LogitsProcessor
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import warnings

from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class ConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count=0
        self.base_model = base_model
        self.eos_token_id = eos_token_id
        if self.base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids=(B*K,L)，scores=(B*K,V)；mask 必须与 scores 完全同形状。
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
            
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = sent[-self.prefix_index:]
                else:
                    hash_key=sent[-self.count:]
                hash_key = hash_key.tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)

                if len(prefix_allowed_tokens) == 0:
                    warnings.warn(
                        f"No valid tokens found for hash_key {hash_key} at step {self.count}. "
                        f"This indicates the model generated an unexpected token. "
                    )
                    # Force EOS token to end invalid sequence
                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                    continue 
                
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        self.count += 1

        scores = scores + mask
        return scores
