#!/bin/bash
# 阅读导航：
# 阅读顺序：先核对 SFT checkpoint 和 CSV/info/index/item 文件，再看 accelerate 配置和 rl.py:train。
# --num_generations=16 表示每条 prompt 生成 16 个候选；--reward_type ranking 对应 rl.py 的排名奖励选择。
# --beam_search True 启用约束候选生成；--beta 控制参考模型 KL 项；配置文件负责 8 进程 DeepSpeed。
# MiniOneRec 推荐导向 RL 启动脚本：accelerate/deepspeed 调用 rl.py。
# num_generations 必须能整除全局 batch；beam_search 控制是否使用约束 beam。

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --config_file ./config/zero2_opt.yaml \
                                    --num_processes 8 --main_process_port 29503 \
                                    rl.py \
                        --model_path path_to_model \
                        --train_batch_size 64 \
                        --eval_batch_size 128 \
                        --num_train_epochs 2 \
                        --gradient_accumulation_steps 2 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train False \
                        --eval_step 0.0999 \
                        --reward_type ranking \
                        --num_generations 16 \
                        --mask_all_zero False \
                        --dynamic_sampling False \
                        --sync_ref_model True \
                        --beam_search True \
                        --test_during_training False \
                        --temperature 1.0 \
                        --learning_rate 1e-5 \
                        --add_gt False \
                        --beta 1e-3 \
                        --dapo False \
                        --output_dir output_dir \
                        --wandb_run_name wandb_name \
                        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
                        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
done
