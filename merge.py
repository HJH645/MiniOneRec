"""合并多 GPU 评估结果。

阅读入口：merge。每个 shard 是一个 JSON 列表，函数按 cuda_list 顺序把它们
拼接成一个列表；不改变单条结果结构，也不计算指标，指标由 calc.py 完成。
"""

import fire
import pandas as pd
import json
from tqdm import tqdm

def merge(input_path, output_path, cuda_list):
    if type(cuda_list) == int:
        cuda_list = [cuda_list]
    cuda_list = list(cuda_list)
    data = []
    for i in tqdm(cuda_list):
        with open(f'{input_path}/{i}.json', 'r') as f:
            data.extend(json.load(f))
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    fire.Fire(merge)
