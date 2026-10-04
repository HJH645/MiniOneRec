# 阅读导航：
# 这是增强的 RQ-Kmeans 支线；先检查 embedding、预训练 codebook 与 checkpoint 路径。
# 调用 rqkmeans_plus.py 更新量化策略，之后由 generate_indices_plus.sh 输出 item→SID 索引。
# RQ-Kmeans+ 训练脚本：在约束聚类结果上继续生成替代 SID。
python rqkmeans_plus.py \
  --data_path ../data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
  --pretrained_codebook_path ../data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.codebooks_constrained.npz \
  --num_emb_list 256 256 256 \
  --e_dim 2560 \
  --lr 1e-4 \
  --epochs 10000 \
  --batch_size 2048
