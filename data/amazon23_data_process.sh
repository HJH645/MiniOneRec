#!/bin/bash
# 阅读导航：
# 先看原始 Amazon23 数据路径、时间范围和输出目录，再进 amazon23_data_process.py 的主入口。
# 其任务是将 Amazon23 的字段与时间戳整理为下游可读的交互和商品文件；这里还不生成 SID。
# Amazon23 数据预处理启动脚本：调用 amazon23_data_process.py 完成格式转换和切分。

python amazon23_data_process.py \
    --dataset {domain} \
    --metadata_file ../meta_{domain}.jsonl \
    --reviews_file ../{domain}.jsonl \
    --user_k 5 \
    --st_year 2018 \
    --st_month 10 \
    --ed_year 2023 \
    --ed_month 9 \
    --output_path ./Amazon23
