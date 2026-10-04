# 阅读导航：
# 先确认增强模型 checkpoint 和 embedding 路径相互匹配。
# 调用 generate_indices_plus.py 生成并去重 SID，输出格式需与 convert_dataset.py 所需 index.json 一致。
# RQ-Kmeans+ SID 生成脚本：读取 embedding 和 codebook，输出 item 到 SID 的索引。
python generate_indices_plus.py \
  --data_path ../data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
  --ckpt_path your_best_collision_model_path: e.g. /Nov-20-2025_12-25-13/best_collision_model.pth \
  --num_emb_list 256 256 256 \
  --device cuda:0
