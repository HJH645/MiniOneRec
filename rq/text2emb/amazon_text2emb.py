"""把商品标题和描述编码成可供 RQ-VAE 使用的 dense embedding。

阅读入口：preprocess_text -> generate_item_embedding。tokenizer 输入是
input_ids=(B,T)、attention_mask=(B,T)，Transformer 输出 last_hidden_state
=(B,T,D)，masked mean pooling 后变成 mean_output=(B,D)，所有商品拼接保存
为 .npy 的 (N,D)；当前样例是 (3686,2560)。

从末尾 __main__ 的参数与模型加载看起，再跟 preprocess_text → load_data →
generate_text → generate_item_embedding。前半段读取 .item.json 并拼 title 与
description；后半段按进程切分商品、按 batch tokenize，经编码器输出
(B,T,D)，用 attention_mask 做 masked mean pooling 得到 (B,D)。
阅读末尾 gather/排序/保存逻辑时检查商品 id 顺序是否与 .item.json 一致；
这个顺序是后续 RQ 索引与商品 id 对齐的前提。
"""

import argparse
import collections
import json
import os
import random
import torch
from tqdm import tqdm
import numpy as np
from utils import * 
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
from accelerate.utils import gather_object

def load_data(args):
    if args.root:
        print("args.root: ", args.root)
    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)
    return item2feature

def generate_text(item2feature, features):
    """按 item.json 的每个商品 id 拼接指定文本字段。

    返回 [(整数 item_id, 清洗后的文本)]；缺少可用字段时填 unknown item。
    """
    item_text_list = []
    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                cleaned = meta_value.strip()
                if cleaned != "":
                    text.append(cleaned)

        if len(text) == 0:
            text = ["unknown item"]
        
        try:
            item_id = int(item)
        except:
            item_id = item
            
        item_text_list.append((item_id, " ".join(text)))

    return item_text_list

def preprocess_text(args):
    print('Process text data: ')
    print('Dataset: ', args.dataset)
    item2feature = load_data(args)
    item_text_list = generate_text(item2feature, ['title', 'description'])
    return item_text_list

def generate_item_embedding(args, item_text_list, tokenizer, model, accelerator, word_drop_ratio=-1):
    """把商品文本批量编码成商品级向量，并保存为 `.npy` 文件。

    参数约定：

    - `item_text_list` 是 `[(item_id, text), ...]`，每个元素代表一个商品；
    - `tokenizer` 把字符串转成 `input_ids` 和 `attention_mask`；
    - `model` 是文本编码器，不是 MiniOneRec 后续训练的语言模型；
    - `accelerator` 提供进程编号、设备和多进程同步/汇总能力；
    - `word_drop_ratio` 大于 0 时，在编码前随机删除文本中的部分空格分词。

    处理结果的形状变化是：

    ```text
    B 条商品文本
      → tokenizer: input_ids/attention_mask=(B,T)
      → 文本编码器: last_hidden_state=(B,T,D)
      → attention mask 加权平均: 商品向量=(B,D)
      → 所有商品按 item_id 排序: final_embeddings=(N,D)
    ```

    这里的 `T` 是当前 batch padding 后的 token 长度，`D` 是文本编码器的
    hidden size，`N` 是商品总数。padding token 不应参与商品向量，所以池化
    时必须使用 `attention_mask`；最后按 `item_id` 排序则是为了保证数组第
    `i` 行仍对应整数商品 `i`，供 RQ-VAE 和 `generate_indices.py` 继续使用。
    """
    # item_text_list 的每一项是 (商品 id, 商品文本)。商品 id 不能丢失：多进程
    # 汇总后要依靠它恢复全局顺序，而不是依赖各进程完成任务的先后顺序。
    all_ids, all_texts = zip(*item_text_list)

    # N 是商品总数；后面的分片按这个数量计算，不按文本字符数计算。
    total_items = len(all_texts)

    # Accelerator 为每个进程提供：总进程数 num_processes，以及当前进程的
    # process_index。所有进程共同处理一份商品列表，每个进程只处理其中一段。
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    # 使用向上取整，让每个分片最多包含 chunk_size 个商品。
    # 例：N=10、进程数=3 时 chunk_size=4，分片范围大致为 [0:4]、[4:8]、[8:10]。
    chunk_size = int(np.ceil(total_items / num_processes))
    start_idx = process_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_items)

    # 当前进程的商品 id 与文本保持一一对应；切片只按商品列表位置进行，
    # 不会改变单个商品的文本内容。
    local_ids = all_ids[start_idx:end_idx]
    local_texts = all_texts[start_idx:end_idx]

    if accelerator.is_main_process:
        print(f"Total items: {total_items}")
        print(f"Start generating embeddings with {num_processes} processes...")

    # 每项保存为 (item_id, embedding)，先保留 id，最后统一排序。
    local_results = []
    # 这是推理 batch size，不是训练 batch size；显存不足时应减小它。
    batch_size = 1024

    # 进度条只在本机主进程显示，避免多 GPU 时多个进度条互相覆盖。
    pbar = tqdm(total=len(local_texts), desc=f"Proc {process_index}", disable=not accelerator.is_local_main_process)

    # 右侧 padding 与 attention_mask 配合使用：真实 token 在左侧保持原顺序，
    # padding 位置会被 mask 掉，因此不会进入后面的平均池化。
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        # 有些 decoder-only tokenizer 没有单独的 pad token；使用 EOS 作为 padding
        # 的占位符，但它是否参与计算由 attention_mask 决定。
        tokenizer.pad_token = tokenizer.eos_token

    # 只做推理，不需要保存反向传播中间结果，可以降低显存占用。
    with torch.no_grad():
        for i in range(0, len(local_texts), batch_size):
            # 当前 batch 的文本和商品 id 必须使用相同切片，zip 时才能正确配对。
            batch_texts = list(local_texts[i : i + batch_size])
            batch_ids = local_ids[i : i + batch_size]

            # 可选的数据增强：按空格切分文本并随机丢词。默认 -1，不做丢词；
            # 该操作只影响当前 batch 的输入，不修改原始 item_text_list。
            if word_drop_ratio > 0:
                processed_batch = []
                for text in batch_texts:
                    sent = text.split(' ')
                    new_sent = [wd for wd in sent if random.random() > word_drop_ratio]
                    processed_batch.append(' '.join(new_sent))
                batch_texts = processed_batch

            # tokenizer 会对一个 batch 自动补齐到相同长度 T：
            # input_ids.shape=(B,T)，attention_mask.shape=(B,T)。超过 max_sent_len
            # 的文本被截断；padding=True 只补齐当前 batch，不保证所有 batch 的 T 相同。
            encoded_sentences = tokenizer(
                batch_texts, 
                max_length=args.max_sent_len,
                truncation=True, 
                return_tensors='pt', 
                padding=True
            ).to(accelerator.device)

            input_ids = encoded_sentences.input_ids
            attention_mask = encoded_sentences.attention_mask

            # 文本编码器逐 token 输出隐藏状态：
            # input_ids=(B,T) → last_hidden_state=(B,T,D)。
            # B 是 batch 商品数，T 是 token 数，D 是编码器隐藏维度。
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # 取编码器的逐 token 隐藏状态。一个商品最终需要一个向量，因此接下来
            # 要沿 token 维（dim=1）聚合，而不是把所有 token 向量直接展平。
            last_hidden = outputs.last_hidden_state

            # attention_mask 原本是 (B,T)，1 表示真实 token，0 表示 padding。
            # unsqueeze(-1) 变成 (B,T,1)，再 expand 成 (B,T,D)，这样每个 token
            # 的 mask 会复制到它的 D 个隐藏维度上，可以与 last_hidden 相乘。
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()

            # 先把 padding 的隐藏状态归零，再沿 token 维求和，得到 (B,D)。
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)

            # 统计每个商品真实 token 的数量，形状为 (B,D)；expand 后每个 hidden
            # 维度使用同一个 token 数。clamp 防止异常空文本导致除零。
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

            # masked mean pooling：(B,D) / (B,D) → (B,D)。每一行就是一个商品向量。
            mean_output = sum_embeddings / sum_mask

            # GPU Tensor → CPU NumPy，避免把所有 batch 的结果长期占在 GPU 显存中。
            mean_output = mean_output.cpu().numpy()

            # 保留商品 id 与向量的绑定关系；当前进程完成后再交给 gather_object。
            for idx, emb in zip(batch_ids, mean_output):
                local_results.append((idx, emb))

            pbar.update(len(batch_texts))
    
    pbar.close()

    # 所有进程都要到达这里，确保每个进程已经完成自己的局部推理。
    accelerator.wait_for_everyone()

    # gather_object 收集各进程的 Python 列表，主进程得到多个 local_results；
    # 由于各进程速度可能不同，此时列表顺序不一定是商品 id 顺序。
    all_results_flat = gather_object(local_results)

    if accelerator.is_main_process:
        print("Gathering finished. Sorting and saving...")

        # 恢复全局商品顺序。若 item_id 是整数，第 i 行才能对应 item_id=i；
        # 这是后续 generate_indices.py 写出 index.json 的必要前提。
        all_results_flat.sort(key=lambda x: x[0])

        # 去掉每项的商品 id，只保留向量并堆叠成二维矩阵 (N,D)。
        final_embeddings = np.stack([x[1] for x in all_results_flat], axis=0)

        print('Final Embeddings shape: ', final_embeddings.shape)

        # 保存路径：args.root/类目.emb-模型名-td.npy。这个文件会被 rqvae.py
        # 作为训练输入，或被 generate_indices.py 作为推理输入。
        file_path = os.path.join(args.root, f"{args.dataset}.emb-{args.plm_name}-td.npy")
        np.save(file_path, final_embeddings)
        print(f"Saved to {file_path}")

def load_qwen_model(model_path):
    print("Loading Qwen Model:", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beauty', help='Beauty / Sports / Toys')
    parser.add_argument('--root', type=str, default="")
    # parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--plm_name', type=str, default='qwen')
    parser.add_argument('--plm_checkpoint', type=str, default='xxx', help='Qwen model path')
    parser.add_argument('--max_sent_len', type=int, default=2048)
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print(f"Running with {accelerator.num_processes} processes.")

    item_text_list = preprocess_text(args)

    plm_tokenizer, plm_model = load_qwen_model(args.plm_checkpoint)
    
    plm_model = plm_model.to(accelerator.device)
    plm_model.eval()

    generate_item_embedding(
        args, 
        item_text_list, 
        plm_tokenizer, 
        plm_model, 
        accelerator, 
        word_drop_ratio=args.word_drop_ratio
    )
