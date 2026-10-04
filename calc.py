"""根据 evaluate.py 输出的候选 JSON 计算离线推荐指标。

阅读入口：主函数读取每条样本的真实 item_sid 和 predict 列表，再按候选排名
计算 HR@K 与 NDCG@K。这里不重新调用模型，输入输出都是 Python 列表/标量，
因此适合在 CPU 上单独检查。

从 gao 开始：读取 info 文本建立合法商品集合，遍历评估 JSON 的目标 SID
和 predict 候选，查命中位置并累积 HR@K、NDCG@K。先手算一条候选的
名次与折损增益，再检查代码对非法候选和未命中的处理。
"""

# from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
# import transformers
# import torch
import os
import fire
import math

import json
import pandas as pd
import numpy as np
    
from tqdm import tqdm
def gao(path, item_path):
    """读取评估 JSON 的 output/predict，计算命中率和折损增益。

    path 可为单个或多个结果文件；item_path 是三列 info 文本。每条记录
    从 predict 中取目标首次出现的名次，按 topk_list 累积指标；CC 统计
    遇到的非法候选（在首次命中目标之前）。这里输出的是控制台指标。
    """
    if type(path) != list:
        path = [path]
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    CC=0
        
    
    f = open(f"{item_path}.txt", 'r')
    items = f.readlines()
    # item_names = [ _[:-len(_.split('\t')[-1])].strip() for _ in items]
    item_names= [_.split('\t')[0].strip() for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:   
            item_dict[item_names[i]].append(item_ids[i])
    
    

    result_dict = dict()
    topk_list = [1, 3, 5, 10, 20, 50]
    n_beam = -1
    for p in path:
        result_dict[p] = {
            "NDCG": [],
            "HR": [],
        }
        f = open(p, 'r')
        import json
        test_data = json.load(f)
        f.close()
        
        text = [ [_.strip("\"\n").strip() for _ in sample["predict"]] for sample in test_data]
        
        for index, sample in tqdm(enumerate(text)):
            if n_beam == -1:
                n_beam = len(sample)
                valid_topk = [k for k in topk_list if k <= n_beam]
                ALLNDCG = np.zeros(len(valid_topk))
                ALLHR = np.zeros(len(valid_topk))
            if type(test_data[index]['output']) == list:
                target_item = test_data[index]['output'][0].strip("\"").strip(" ")
            else:
                target_item = test_data[index]['output'].strip(" \n\"")
            # 首次命中的零基排名；未命中保留大哨兵值，所有 Top-K 均记 0。
            minID = 1000000
            for i in range(len(sample)):
                
                if sample[i] not in item_dict:
                    CC += 1
                    print(sample[i])
                    print(target_item)
                if sample[i] == target_item:
                    minID = i
                    break
            
            for index, topk in enumerate(topk_list):
                if topk > n_beam:
                    continue
                if minID < topk:
                    ALLNDCG[index] = ALLNDCG[index] + (1 / math.log(minID + 2))
                    ALLHR[index] = ALLHR[index] + 1
        print(n_beam)
        valid_topk = [k for k in topk_list if k <= n_beam]
        print(valid_topk)
        print(f"NDCG:\t{ALLNDCG / len(text) / (1.0 / math.log(2))}")
        print(f"HR\t{ALLHR / len(text)}")
        print(CC)

if __name__=='__main__':
    fire.Fire(gao)
