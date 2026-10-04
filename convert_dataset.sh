#!/bin/bash
# 阅读导航：
# 先确认 INPUT_DIR 内同名 .item.json、.index.json、.train/.valid/.test.inter 齐全。
# 再检查 OUTPUT_DIR、类别及命令参数；脚本调用 convert_dataset.py 将整数商品 id 对齐为 SID。
# 输出的 train/valid/test CSV 供 data.py 读取，info 文本供约束解码和指标计算。
# 数据格式转换脚本：把 RQ 产出的 item/index/inter 文件转换为训练 CSV 和 info 文件。
# 把 RQ 产生的 item/index/inter 文件转换为 MiniOneRec 的 train/valid/test CSV。
# 入口实现是 convert_dataset.py:main。


PYTHON_SCRIPT="convert_dataset.py"

INPUT_DIR="data/Amazon18/Industrial_and_Scientific"

OUTPUT_DIR="data/Amazon18"

DATASET_NAME="Industrial_and_Scientific"

# ===========================================

echo "Start converting $DATASET_NAME ..."

python $PYTHON_SCRIPT \
    --dataset_name $DATASET_NAME \
    --data_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --category $DATASET_NAME \
    --seed 42

echo "Finished!"
