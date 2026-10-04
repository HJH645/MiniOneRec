#!/bin/bash
# 阅读导航：
# TS-Rec 支线入口；先核对 train/valid CSV、SID 索引和描述关键词文件。
# 然后按参数进入 ts_rec_sft.py:train；与主线 sft.sh 比较额外任务和新增 SID token 的初始化方式。
# TS-Rec SFT 启动脚本：服务 TS-Rec 支线，参数和主线 sft.sh 不完全相同。
export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

DATASET="Industrial_and_Scientific"
# DATASET="Office_Products"
# DATASET="Toys_and_Games"

MODEL="Qwen2.5-1.5B"

for category in $DATASET; do
    
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    torchrun --nproc_per_node 8 \
            sft.py \
            --base_model ${MODEL} \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ${MODEL}_ts_rec_sft_${category} \
            --wandb_project MiniOneRec_SFT \
            --wandb_run_name ${MODEL}_ts_rec_sft_${category} \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/${category}.index.json \
            --item_meta_path ./data/Amazon/index/${category}.item.json \
            --freeze_LLM False \
            --description_path ./data/Amazon/${category}.description_keywords_rqvae.json \
            --learning_rate 3e-4
done