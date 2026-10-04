# MiniOneRec 中文源码导学

> 目标：用一条可验证的主线读懂 MiniOneRec：**商品文本 → 语义 ID（SID）→ 语言模型生成 → 约束解码 → 推荐指标**。\
> 所有路径都相对于仓库根目录 `MiniOneRec/`。第一次学习只走主线，支线放到最后对照。

## 1. 先看全局流程

```text
原始 Amazon reviews / metadata
  │
  ├─ data/amazon18_data_process.py       清洗、过滤、按时间切分交互
  │       ↓
  ├─ rq/text2emb/amazon_text2emb.py      title + description → 商品 embedding
  │       ↓
  ├─ rq/rqvae.py + rq/models/*           embedding → 多层量化索引
  │       ↓
  ├─ rq/generate_indices.py              量化索引 → item-SID index.json
  │       ↓
  ├─ convert_dataset.py                  item/index/interaction → train/valid/test CSV
  │       ↓
  ├─ sft.py + data.py                    历史 SID → 目标 SID（监督微调）
  │       ↓
  ├─ rl.py + minionerec_trainer.py       多候选生成 + 推荐奖励（GRPO）
  │       ↓
  └─ evaluate.py + calc.py               约束生成 → HR@K / NDCG@K
```

不要从 `minionerec_trainer.py` 第一行开始。先完成数据格式和 SID，再回到训练器。

## 2. 学习方式和阶段验收

每个阶段都按“**先读入口 → 再跟一条样例 → 最后回答问题**”进行。没有 GPU 时只做静态阅读、shape 推导和小数据检查，不直接运行默认训练脚本。

| 阶段 | 目的 | 先读文件 | 产物/验收 |
| --- | --- | --- | --- |
| 0. 环境与地图 | 知道脚本入口和依赖 | `README.md`、`requirements.txt`、`sft.sh`、`rl.sh`、`evaluate.sh` | 能说出三阶段训练/评估分别由谁启动 |
| 1. 交互数据 | 明白一行 CSV 如何产生 | `data/amazon18_data_process.py`、`convert_dataset.py`、`data.py` | 能解释 history、target、SID 三者关系 |
| 2. SID 构造 | 明白文本为何变成离散 token | `rq/text2emb/amazon_text2emb.py`、`rq/rqvae.py`、`rq/models/rqvae.py`、`rq/models/rq.py`、`rq/models/vq.py` | 能写出 `(B,T,D)→(B,D)→(B,3)` 的 shape 链 |
| 3. SFT | 明白 prompt 和 loss | `sft.py`、`data.py:SidSFTDataset` | 能解释 labels 中 `-100` 的作用 |
| 4. 约束生成 | 明白 SID 为什么合法 | `evaluate.py`、`LogitProcessor.py` | 能解释 prefix map 如何屏蔽非法 token |
| 5. RL | 明白 reward 如何更新模型 | `rl.py`、`minionerec_trainer.py`、`sasrec.py` | 能解释 G 个候选、advantage 和 KL |
| 6. 评估 | 明白结果如何变成指标 | `split.py`、`evaluate.py`、`merge.py`、`calc.py` | 能从 JSON 的 output/predict 找到目标排名并计算 HR/NDCG |
| 7. 支线 | 比较变体和传统基线 | `*_gpr.py`、`ts_rec_*.py`、`rq/rqkmeans*`、`utility.py` | 能说出支线改变了哪一层 |

## 3. 前置知识（按优先级）

| 知识点 | 在项目中的位置 | 优先级 |
| --- | --- | --- |
| Python 函数、类、继承、`__getitem__` | `data.py`、`minionerec_trainer.py` | 必须 |
| 列表、字典、集合、切片、`eval` | `data.py`、`LogitProcessor.py` | 必须 |
| JSON/CSV 和 Pandas | `convert_dataset.py`、`data.py` | 必须 |
| NumPy 与 PyTorch Tensor | `rq/datasets.py`、`rq/models/*` | 必须 |
| Dataset/DataLoader 与 batch | `data.py`、`rq/trainer.py` | 必须 |
| Causal LM、交叉熵、next-token prediction | `sft.py` | 重要 |
| Beam search、logits mask | `evaluate.py`、`LogitProcessor.py` | 重要 |
| GRPO、advantage、KL | `rl.py`、`minionerec_trainer.py` | 重要 |
| Accelerate、DeepSpeed、多 GPU | `rl.sh`、`config/zero2_opt.yaml` | 后学 |

## 4. 第 0 阶段：入口和目录地图

阅读顺序：

1. `README.md` 的 Key Techniques、Repository Overview、Full Pipeline Walk-through。
2. `sft.sh`：确认模型、CSV、SID 索引和 batch 参数如何传给 `sft.py`。
3. `rl.sh`：确认 `num_generations`、`beam_search`、`beta` 等参数。
4. `evaluate.sh`：确认 `split → evaluate → merge → calc` 的调用顺序。
5. `requirements.txt`：把 Transformers、TRL、Accelerate、DeepSpeed、PyTorch 与阶段对应起来。

验收：不用看代码，能画出“数据准备、SID 构造、SFT、RL、评估”的五个框。

## 5. 第 1 阶段：交互如何变成训练样本

主链：

```text
data/amazon18_data_process.py
  → 读取评论和元数据 → 迭代 K-core 过滤
  → 用户/商品整数 id 映射 → 按用户滑窗构造最多 10 项历史
  → 按全局目标时间 8:1:1 切分 .inter 文件，并写出 .item.json
rq/text2emb/amazon_text2emb.py
  → 读取 .item.json 的 title/description → 商品 embedding .npy
rq/rqvae.py + rq/models/*
  → 训练 RQ-VAE checkpoint
rq/generate_indices.py
  → embedding 行号/整数 item_id → SID → .index.json
convert_dataset.py
  → 读取同一类目的 .item.json + .index.json + .inter
  → history_item_sid / item_sid 的 CSV
data.py:CSVBaseDataset → SidSFTDataset.pre
  → instruction + history prompt + target SID
```

建议跟一条评论：`reviewerID` 和 `asin` 先被映射为整数 id；`unixReviewTime` 决定它在用户历史中的位置；一条滑窗样本最终写成 `.inter` 的“用户 id、历史 id 列表、目标 id”。这里还没有 SID；`amazon18_data_process.py` 只负责生成 `.inter/.item.json`，SID 要等 RQ-VAE 训练后由 `rq/generate_indices.py` 生成 `.index.json`，再由 `convert_dataset.py` 拼起来。注意当前 K-core 调用只使用 `user_k`，`item_k` 参数没有传入过滤函数。

### 你刚才指出的关键连接：item_id → SID 在哪里做？

完整链路不是“预处理直接得到 SID”，而是三段文件在不同阶段产生：

| 阶段 | 负责文件 | 产生什么 |
| --- | --- | --- |
| 1. 交互预处理 | `data/amazon18_data_process.py` | `.inter`、`.item.json`、`.item2id`；其中 `item_id` 是连续整数 |
| 2. 文本/RQ | `rq/text2emb/amazon_text2emb.py`、`rq/rqvae.py` | `.emb-*.npy` 和 RQ-VAE checkpoint |
| 3. SID 生成 | `rq/generate_indices.py` | `.index.json`：`整数 item_id → [<a_i>, <b_j>, <c_k>]` |
| 4. 格式转换 | `convert_dataset.py` | 读取上述三类文件，写 `history_item_sid/item_sid` CSV 和 `info/*.txt` |

`rq/generate_indices.py` 里的这段代码就是映射的落点：

```python
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)
```

这里的 `item` 不是重新从评论里查出来的 ASIN，而是 embedding 的行号。这个设计要求：生成 embedding 时，商品顺序必须和 `.item.json` 的整数 `item_id` 顺序一致。当前 `amazon_text2emb.py` 会按商品 id 排序后保存 embedding，所以第 118 行对应整数商品 `118`，生成的 `.index.json` 才能被 `convert_dataset.py` 正确查到。

因此，`convert_dataset.py` 的输入目录必须同时包含：

```text
Industrial_and_Scientific.item.json       # 商品整数 id → 标题/描述
Industrial_and_Scientific.index.json      # 商品整数 id → SID
Industrial_and_Scientific.train.inter     # 用户、历史整数 id、目标整数 id
Industrial_and_Scientific.valid.inter
Industrial_and_Scientific.test.inter
```

当前仓库实际把已转换结果放在 `data/Amazon/`，而 `convert_dataset.sh` 仍写着旧示例路径 `data/Amazon18/`，所以直接运行这个 shell 脚本会找不到目录；它需要按你的实际输入目录修改，不能据此推断 `.index.json` 是 `amazon18_data_process.py` 生成的。

### `metadata`、`metadata_file`、`id_title` 到底是什么

这里的 `metadata` 指 **Amazon 原始商品元数据**，不是模型中间结果。原始文件通常是一行一个 JSON 商品对象：

```json
{"asin": "B001ABC", "title": "商品标题", "description": ["商品描述"], "brand": "品牌"}
```

`metadata_file` 是这个文件的路径，来自命令行参数：

```bash
python amazon18_data_process.py \
  --metadata_file ../meta_Industrial_and_Scientific.json
```

如果不传 `--metadata_file`，代码默认使用 `../meta_{category}.json`。这个相对路径是相对于“启动 Python 时的当前目录”，不是相对于 `amazon18_data_process.py` 文件本身：从 `data/` 目录运行时会找仓库根目录下的 `meta_*.json`；从仓库根目录直接运行时会找仓库上一级。当前仓库没有原始 `meta_*.json` 和 review JSONL，所以不能直接从头重跑这一步；仓库里的 `data/Amazon/index/*.item.json` 已经是处理后的“整数 item_id → 商品特征”对象，格式不同，不能直接替代这里的 metadata。

`id_title` 不是输入文件，也不是外部传入的变量。它在 `load_metadata_json2csv_style` 内部创建：

```python
id_title = {}
id_title[meta["asin"]] = meta["title"]
```

只有标题存在、标题长度大于 1 且单词数不超过 20 的商品才会进入这个字典。因此它的实际类型是：

```text
原始 ASIN  →  清洗后的商品标题
B001ABC    →  "某个商品标题"
```

调用关系是：`main` 调 `process_dataset_recursive`；后者虽然接收了一个叫 `metadata` 的参数，但当前调用传的是 `None`，函数内部真正执行 `load_metadata_json2csv_style(args.dataset, args.metadata_file)`；该函数一次返回 `metadata、id_title、remove_items`。之后 `id_title` 被传给 K-core 过滤，检查 review 里的 `asin` 是否有可用标题；再传给滑窗函数，为历史商品填 `titles`；最后完整 `metadata` 和 `id_title` 一起用于生成 `.item.json`。

### 先把 `asin` 和 K-core 讲明白

Amazon 原始数据里的 `asin` 是 **Amazon Standard Identification Number**，也就是商品编号。比如一条 review 可以抽象成：

```text
reviewerID = 用户 A
asin       = 商品 X
unixReviewTime = 某个时间
overall    = 5
```

这表示“用户 A 在这个时间评价了商品 X”。`reviewerID` 是用户节点，`asin` 是商品节点；`reviewText`、`summary` 和 `overall` 是这条交互的附加信息。后续 `convert_inters2dict_amazon18_style` 才把原始 `reviewerID/asin` 映射成连续整数 `user_id/item_id`，所以不要把 `asin` 和 CSV 中的整数 `item_id` 混为一谈。

`k_core_filtering_json2csv_style` 可以先当成一个“清理交互图”的函数：review 是连接用户和商品的一条边，`K=5` 表示留下来的每个用户至少有 5 条边、每个商品至少有 5 条边。函数的阅读顺序是：

1. 看 `remove_users`、`remove_items`：它们记录已经被淘汰的用户和商品。
2. 看第一段 `for review in reviews`：没有合法标题的商品先加入 `remove_items`。
3. 看 `while True` 里面读取每条 review：先按时间窗口过滤，再跳过已淘汰的用户/商品，然后分别给 `user_counts[reviewerID]` 和 `item_counts[asin]` 加一。
4. 看两个 `for` 循环：频次 `< K` 的用户或商品被标记，`flag=True` 表示还要继续下一轮。
5. 看 `reviews = new_reviews`：下一轮只在本轮剩余的 review 上统计，直到 `flag=False`。

为什么要循环？假设 `K=2`：用户 A 评价 X、Y，用户 B 只评价 X，用户 C 评价 X、Z。第一轮 Y、Z 各只有 1 条，B 只有 1 条，被删除；第二轮 A、C 都只剩 X 这一条，也被删除；第三轮 X 也没有用户留下。这种“删除一个对象导致另一个对象变低频”的连锁反应，就是 K-core 过滤。

这段函数的输出不是 SID，而是仍然含 `reviewerID`、`asin`、时间和评分的过滤后 review 列表。只有后面的 `convert_inters2dict_amazon18_style`、`generate_interaction_list_json2csv_style` 才把它变成推荐样本。

仓库里的一个可核对样本位于 `data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv` 首行：`history_item_id=[117]`、`item_id=118`、`history_item_sid=['<a_165><b_107><c_44>']`、`item_sid='<a_104><b_118><c_176>'`。打开 `data/Amazon/index/Industrial_and_Scientific.index.json`，键 `"118"` 的值正是 `['<a_104>', '<b_118>', '<c_176>']`；再打开同名 `.item.json`，键 `"118"` 有目标商品的标题。这样可以先验证 **CSV → 索引 → 商品元数据** 的对应关系。原始 Amazon review 文件不在仓库中，向前追溯到 `reviewerID/asin/unixReviewTime` 要结合预处理源码，不能从这条 CSV 反推出原始评论。

重点符号：

- 先从 `data/amazon18_data_process.py` 文件末尾的 `if __name__ == "__main__"` 看起：`load_reviews_json2csv_style` 读评论，`process_dataset_recursive` 读取商品元数据并调用 `k_core_filtering_json2csv_style` 迭代过滤；随后 `convert_inters2dict_amazon18_style` 编整数 id，`generate_interaction_list_json2csv_style` 按用户构造最多 10 项历史，再由 `convert_to_atomic_files_json2csv_style` 按全局目标时间做 8:1:1 切分。`data/process.py:gao` 是旧版对照。
- `convert_dataset.py:load_dataset`、`semantic_tokens_to_id`、`convert_interactions_to_csv`：看 item id 与 SID 如何对齐。
- `data.py:CSVBaseDataset`、`SidSFTDataset.get_history`、`SidSFTDataset.pre`：看 CSV 字符串如何恢复成列表并拼成 prompt。

shape 只记三层：一行 CSV → `input/output: str` → tokenizer 后 `input_ids/attention_mask/labels: (L,)`；批处理后为 `(B,L_batch)`。CSV 中的列表是字符串，原代码使用 `eval`，不可信数据应改为 `ast.literal_eval`。

验收问题：历史长度为 `H=10` 时，为什么 tokenizer 后长度不一定是 30？

## 6. 第 2 阶段：文本向量和 SID

### 先用一句话理解 RQ-VAE

RQ-VAE 的任务是把一个连续商品向量压缩成少量整数编号。可以把每层 codebook 想成一本“原型向量字典”：`VectorQuantizer` 从字典里找一个最接近当前向量的原型；`ResidualVectorQuantizer` 第一层找完后，把已经解释的部分减掉，再让第二层解释剩下的误差，最后把各层编号拼成 SID。

例如一个商品经过 encoder 得到 `z=(0.8, 0.1)`：

```text
第 1 层：选中 code <a_12>，向量 q1=(0.7, 0.0)
残差：z-q1=(0.1, 0.1)
第 2 层：选中 code <b_4>，向量 q2=(0.1, 0.08)
残差：(0.0, 0.02)
第 3 层：选中 code <c_9>，继续解释剩余误差
最终 SID：<a_12><b_4><c_9>
最终近似向量：q1+q2+q3
```

因此 `ResidualVectorQuantizer.forward` 里的 `residual = residual - x_res` 不是丢弃信息，而是在把已经编码的部分扣掉；`x_q = x_q + x_res` 则是在累计最终的离散近似。每层返回一个 `(B,)` 的编号，最后 `torch.stack(..., dim=-1)` 得到 `(B,L)`，这就是后面写入 `.index.json` 的 SID 编号矩阵。

`RQVAE` 外层只负责连接三段：`encoder` 把 `(B,2560)` 压成 `(B,32)`，`ResidualVectorQuantizer` 把 `(B,32)` 变成离散编号和量化向量，`decoder` 再把量化向量还原为 `(B,2560)`。训练时同时要求“还原得像原 embedding”和“codebook/encoder 彼此靠近”；推理生成 SID 时只调用 `get_indices`，不需要 decoder。

### RQ 代码推荐阅读顺序

1. `rq/datasets.py:EmbDataset`：确认 `.npy` 第 `i` 行是一个 `(D,)` 商品向量。
2. `rq/models/layers.py:MLPLayers.forward`：理解 encoder/decoder 只是按维度列表堆叠的 MLP。
3. `rq/models/vq.py:VectorQuantizer.forward`：先理解一个 codebook 如何计算距离、选索引、查表和计算量化损失。
4. `rq/models/rq.py:ResidualVectorQuantizer.forward`：把上一层的 residual 传给下一层，并累计 `x_q`。
5. `rq/models/rqvae.py:RQVAE.forward/get_indices/compute_loss`：把 encoder、RQ、decoder 和两类损失连起来。
6. `rq/trainer.py:_train_epoch/_valid_epoch`：看训练如何调用模型，以及如何用重复索引计算 collision rate。
7. `rq/generate_indices.py`：确认 `(B,L)` 的索引如何格式化成 `<a_i><b_j><c_k>` 并写成 `item_id → SID`。

主链：

```text
rq/text2emb/amazon_text2emb.py:generate_item_embedding
  input_ids=(B,T), last_hidden_state=(B,T,D)
  → attention mask 加权平均 → embedding=(B,D)
  → 保存为 .npy（当前样例为 (3686,2560)）
rq/models/rqvae.py:RQVAE
  → encoder: (B,2560)→(B,32)
  → rq/models/rq.py:ResidualVectorQuantizer
  → 三层 VectorQuantizer → indices=(B,3)
  → decoder 重建 (B,2560)
```

层数不是固定常数，来自 `--num_emb_list` 的长度；改变层数会同时改变 SID 长度和约束表。阅读顺序是 `rq/datasets.py:EmbDataset` → `rq/rqvae.py` → `rq/models/rqvae.py` → `rq/models/rq.py` → `rq/models/vq.py` → `rq/generate_indices.py`。

验收问题：为什么量化器每层都处理 residual？`index.json` 中的三个 SID 片段分别从哪里来？

## 7. 第 3 阶段：SFT 学习下一个 SID

主链：`sft.sh → sft.py:train → TokenExtender → SidSFTDataset/SidItemFeatDataset/FusionSeqRecDataset → Trainer`。

单条样本：

```text
instruction=(I,) + history prompt=(P,) + target SID=(Y,)
→ input_ids/attention_mask/labels=(L,)
→ batch=(B,L_batch)
→ logits=(B,L_batch,V)
```

`labels = [-100] * (I+P) + target_tokens`，因此 prompt 不计入交叉熵，只有目标 SID 和 EOS 提供监督。扩展词表后 embedding 从 `(V_old,H)` 变为 `(V_old+V_sid,H)`；`freeze_LLM=True` 时只更新新增 SID 行。

验收问题：如果目标 SID 被截断，哪些 label 还会参与 loss？如何从日志确认词表扩展生效？

## 8. 第 4 阶段：约束解码

主链：`info/*.txt` 的完整 SID → tokenizer → `hash_dict`（前缀到合法下一个 token）→ `ConstrainedLogitsProcessor` → beam search。

有 `K` 个 beam 时，Transformers 将输入展平为 `input_ids=(B*K,L)`、`scores=(B*K,V)`；处理器把非法位置设为 `-inf`，输出 shape 不变。`prefix_index=3` 与 prompt 模板、tokenizer 有关，换模型必须重新验证。每次 `generate` 都要创建新的 processor，因为 `count` 会递增。

验收问题：如果当前前缀不存在于 hash 表，代码如何处理？为什么 `do_sample=False` 对评估很关键？

## 9. 第 5 阶段：RL/GRPO

主链：

```text
rl.py:train → Dataset → ReReTrainer
→ 每条 prompt 生成 G 个 completion
→ rule / ndcg / semantic / SASRec reward
→ reward.view(-1,G) → advantage
→ 当前策略 log-prob + reference KL → loss
```

令本地 prompt 数为 `B`、候选数为 `G`、prompt/completion 长度为 `P/C`：`prompt_ids=(B*G,P)`、`completion_ids=(B*G,C)`、`per_token_logps=(B*G,C)`、`reward=(B*G,)`、`advantage=(B*G,)`。`minionerec_trainer.py:compute_loss` 中先看默认分支，再看 `dapo/gspo`；`sasrec.py:SASRec.forward_eval` 只在启用协同过滤奖励时进入。

验收问题：组内标准化为什么需要 `G` 个候选？KL 项限制了什么变化？

## 10. 第 6 阶段：评估闭环

```text
evaluate.sh
  → split.py：测试 CSV 切 shard
  → evaluate.py：每个 shard 约束生成 JSON
  → merge.py：按 shard 合并
  → calc.py：查目标排名，计算 HR@K/NDCG@K
```

重点确认：候选 JSON 的 `predict`、真实 `output` 是否和 test CSV 的 `item_sid` 使用同一套 SID 字符串；非法商品数量（`CC`）异常时，先检查 Transformers 版本、tokenizer 和 prefix 表。

验收问题：候选有重复时 rank 如何处理？为什么要先 merge 再算指标？

## 11. 无 GPU 的最小实验

### 实验 A：检查 CSV

```bash
python - <<'PY'
import pandas as pd
p = 'data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv'
df = pd.read_csv(p)
print(df.shape)
print(df.iloc[0][['history_item_sid', 'item_sid']])
PY
```

### 实验 B：检查 embedding

```bash
python - <<'PY'
import numpy as np
x = np.load('data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy')
print(x.shape, x.dtype)
PY
```

### 实验 C：手算一个 SFT 样本

阅读 `data.py:SidSFTDataset.pre`，记录 `I/P/Y`，验证 `L=min(I+P+Y,max_len)`，并指出哪些 label 为 `-100`。

### 实验 D：手工约束解码

为 `LogitProcessor.py` 构造只有两个合法后继 token 的 `hash_dict`，检查非法位置是否被屏蔽；先做这个小实验，再阅读 50-beam 生成。

## 12. 支线阅读顺序

完成主线后再读：

1. `sft_gpr.py`、`rl_gpr.py`、`convert_dataset_gpr.py`、`data/amazon18_data_process_gpr.py`：看 GPR/VAFT/HEPO 改动哪一层。
2. `ts_rec_data.py`、`ts_rec_sft.py`、`ts_rec_sft.sh`：看新版 TS-Rec 的输入与 prompt 差异。
3. `rq/rqkmeans_faiss.py`、`rq/rqkmeans_constrained.py`、`rq/rqkmeans_plus.py` 及对应 shell：对比 RQ-VAE 的 SID 替代算法。
4. `sasrec.py`、`utility.py`、`SASRecModules_ori.py`：理解传统序列推荐和可选 CF reward。

## 13. 推荐阅读索引

| 主题 | 文件与符号 | 读完能回答什么 |
| --- | --- | --- |
| 项目地图 | `README.md`、`sft.sh`、`rl.sh`、`evaluate.sh` | 主流程从哪里启动？ |
| 数据清洗 | `data/amazon18_data_process.py:process_dataset_recursive` | 一条交互如何进入切分文件？ |
| 数据转换 | `convert_dataset.py:load_dataset`、`convert_interactions_to_csv` | item id 与 SID 如何对齐？ |
| 文本 embedding | `rq/text2emb/amazon_text2emb.py:generate_item_embedding` | `(B,T,D)` 为什么变成 `(B,D)`？ |
| RQ-VAE | `rq/models/rqvae.py`、`rq/models/rq.py`、`rq/models/vq.py` | `(B,2560)` 如何变成 `(B,3)`？ |
| SFT Dataset | `data.py:SidSFTDataset.pre`、`sft.py:train` | 哪些 token 参与 loss？ |
| 约束解码 | `evaluate.py`、`LogitProcessor.py` | 如何避免非法 SID？ |
| GRPO | `rl.py`、`minionerec_trainer.py` | reward 如何进入 loss？ |
| 指标 | `split.py`、`merge.py`、`calc.py` | JSON 如何变成 HR/NDCG？ |

## 14. 常见边界

- JSON 键是字符串，CSV 的 `item_id` 可能是整数，代码经常使用 `str(item_id)`。
- `'<a_1><b_2><c_3>'` 不保证对应 tokenizer 的三个 token，真实长度以 tokenizer 输出为准。
- SFT tokenizer 使用 left padding，文本 embedding 脚本使用 right padding。
- `num_generations` 必须整除全局 batch，训练器会检查。
- 完整训练需要大 GPU、模型 checkpoint 和额外数据；本地样例只适合检查 shape 和格式。
- Transformers/TRL 版本会影响生成参数、`logits_to_keep` 和 DeepSpeed 行为，结论以实际版本和日志为准。

## 15. 学完后的复述题

1. `(2560,)` 商品文本向量为什么能得到三个 SID 片段？
2. 历史长度 `H=10` 为什么不等于 30 个语言模型 token？
3. SFT 的 `labels` 为什么把 prompt 设为 `-100`？
4. beam search 的 `scores` 为什么是 `(B*K,V)`？
5. GRPO 中 `reward.view(-1,G)` 和 `advantage` 各表达什么？
6. `info/*.txt`、`index.json`、CSV 和评估 JSON 的 output/predict 如何保持 SID/item 对齐？

如果答不出来，回到对应阶段的主链和 shape；仍然看不懂时，可以让 AI 按“输入、每一步 shape、输出、一个样例”逐函数带读。

## 16. 文件注释约定

本导学流程中会读到的 Python、Shell 和 YAML 文件都在文件开头写了中文用途说明；注释只描述源码能够证明的行为，不把推测写成事实。主线文件优先看模块头注释和“阅读入口”，再进入函数实现。
