# 阅读导航：
# 从 --data_path 看输入的 (N,D) 商品 embedding，再看 --ckpt_dir 输出 checkpoint。
# 脚本调用 rqvae.py；模型负责 encoder→残差量化→decoder，Trainer.fit 负责训练与碰撞率验证。
# 训练出的 checkpoint 由 generate_indices.py 读取，不能直接当 index.json 使用。
# 训练 RQ-VAE：输入商品 embedding=(N,D)，输出 checkpoint，后续由 generate_indices.py 生成 SID。
python rqvae.py \
      --data_path ../data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
      --ckpt_dir ./output/Industrial_and_Scientific \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480
