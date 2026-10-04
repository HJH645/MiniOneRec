# 阅读导航：
# 阅读顺序：先核对 train/valid CSV 与 index/item JSON，再看 torchrun 参数，最后进入 sft.py:train。
# --batch_size 是目标全局 batch，--micro_batch_size 是每进程一次前向的样本数；训练代码据此计算梯度累积。
# --freeze_LLM 决定是否只更新新增 SID 词表行；示例中的模型、输出和 wandb 路径都需替换。
# MiniOneRec SFT 启动脚本：读取 train/valid/test 和 SID 索引，调用 sft.py。
# 先阅读这里的参数，再进入 sft.py:train；默认需要 8 张 GPU。
export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    torchrun --nproc_per_node 8 \
            sft.py \
            --base_model your_model_path \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_dir/xxx \
            --wandb_project wandb_proj \
            --wandb_run_name wandb_name \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
            --item_meta_path ./data/Amazon/index//Industrial_and_Scientific.item.json \
            --freeze_LLM False
done
