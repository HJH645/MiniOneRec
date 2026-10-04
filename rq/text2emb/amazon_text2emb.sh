# 阅读导航：
# 从 --root 定位商品 .item.json，从 --plm_checkpoint 定位文本编码器。
# accelerate 启动多个进程并行生成商品 embedding；Python 入口是 amazon_text2emb.py 底部。
# 输出 .npy 的商品行顺序必须与后续 SID 生成时读取的商品 id 顺序相符。
# 商品文本向量生成脚本：把 title/description 编码成 .npy embedding，供 RQ 使用。
accelerate launch --num_processes 8 amazon_text2emb.py \
    --dataset Industrial_and_Scientific \
    --root ../../data/Amazon18/Industrial_and_Scientific \
    --plm_checkpoint your_emb_model_path
