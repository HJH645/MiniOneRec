# MiniOneRec 中文源码导学

> 这份文档面向第一次学习 MiniOneRec、并且 Python 基础还不牢固的读者。路径均相对于仓库根目录 `MiniOneRec/`。

## 0. 先记住一句话

MiniOneRec 把“推荐下一个商品”改写成语言模型生成问题：

```text
商品标题/描述 -> 文本向量 -> RQ-VAE 离散语义 ID（SID）
-> 用户历史 SID 序列 -> 语言模型预测下一个 SID
-> 约束解码保证 SID 合法 -> HR@K / NDCG@K
```

推荐主线是：

```text
data/amazon18_data_process.py
  -> rq/text2emb/amazon_text2emb.py
  -> rq/rqvae.py + rq/models/*
  -> rq/generate_indices.py
  -> convert_dataset.py
  -> sft.py + data.py
  -> rl.py + minionerec_trainer.py + LogitProcessor.py
  -> evaluate.py + calc.py
```

不要一开始从 `minionerec_trainer.py` 第一行读起。它是改造过的 TRL/Transformers 训练器，最复杂；先理解数据和 SID，再回来读 RL。

## 1. 范围和源码版本

- 项目：快手 MiniOneRec，当前工作区目录。
- 当前源码版本：工作区最新提交为 `5f4f733`，提交信息为修复冻结 LLM 词表大小。
- 已覆盖：Amazon 数据预处理、文本 embedding、RQ-VAE/RQ-Kmeans SID、数据转换、SFT、GRPO/RL、约束解码、离线评估、SASRec 协同过滤奖励。
- 后读支线：`*_gpr.py`、`ts_rec_*.py`、`rq/*plus*`、传统 `sasrec.py` 的完整训练细节。
- 本机没有执行完整训练：完整流程需要大 GPU、模型 checkpoint 和额外数据；本文的 shape 依据源码和仓库样例推导，具体 batch 大小由启动参数决定。

## 2. 前置知识

| 知识点 | 为什么需要 | 在项目中的位置 | 高频度 |
| --- | --- | --- | --- |
| Python 变量、函数、类、继承 | 看懂 Dataset、Trainer 和模型类 | `data.py`、`minionerec_trainer.py` | 必须 |
| 列表、字典、集合、切片 | 处理 SID、token 和 batch | `data.py`、`LogitProcessor.py` | 必须 |
| JSON、CSV、`with open` | 读取商品、索引和交互 | `convert_dataset.py`、`data.py` | 必须 |
| NumPy 数组 | 读取 embedding 和检查 shape | `rq/datasets.py`、`rq/text2emb/amazon_text2emb.py` | 必须 |
| PyTorch Tensor | 理解模型输入输出 | `rq/models/*.py`、`minionerec_trainer.py` | 必须 |
| Dataset / DataLoader | 理解 batch 如何产生 | `data.py`、`rq/trainer.py` | 必须 |
| Transformer 语言模型 | 理解 logits 和生成 | `sft.py`、`evaluate.py` | 重要 |
| 交叉熵和 next-token prediction | 理解 SFT loss | Transformers Causal LM | 重要 |
| Beam Search 和 logits mask | 理解 SID 合法性约束 | `LogitProcessor.py`、`evaluate.py` | 重要 |
| GRPO | 理解按组归一化奖励 | `rl.py`、`minionerec_trainer.py` | 重要 |
| 多 GPU / Accelerate / DeepSpeed | 理解启动脚本 | `sft.sh`、`rl.sh`、`config/zero2_opt.yaml` | 后学 |

### 给 Python 初学者的最低语法提示

1. `class A(B)` 表示 A 继承 B；例如 `SidSFTDataset(CSVBaseDataset)` 复用 CSV 读取能力。
2. `__init__` 是对象初始化函数；`__getitem__` 决定 `dataset[i]` 返回什么。
3. `self` 是当前对象；`self.data` 是对象内部保存的数据。
4. `dict[key]` 取字典值；`set(...)` 去重；`for x in list` 逐项遍历。
5. `x[:, -N:]` 表示所有样本的最后 N 个位置；`x.view(...)` 只能重排元素，元素总数不能变。
6. `torch.no_grad()` 表示推理时不保存反向传播所需的中间结果；`@torch.no_grad()` 是它的装饰器写法。

## 3. 重点亮点和学习顺序

| 亮点 | 为什么重要 | 通用技术点 | 先看文件 | 顺序 |
| --- | --- | --- | --- | --- |
| 语义 ID 建模 | 商品被压缩成固定长度 token，才能由 LLM 生成 | 离散表示、向量量化 | `rq/models/rqvae.py`、`rq/models/rq.py`、`rq/models/vq.py` | 1 |
| 数据转 prompt | 推荐输入最终是文本和 SID，而不是原始表格 | Dataset、tokenization、padding | `convert_dataset.py`、`data.py` | 2 |
| SFT | 先学习“历史 SID → 下一个 SID” | causal LM、labels、梯度累积 | `sft.py`、`data.py:SidSFTDataset` | 3 |
| 合法候选约束 | 只允许生成商品集合中的 token | prefix map、logits mask、beam | `evaluate.py`、`LogitProcessor.py` | 4 |
| 推荐导向 RL | 从多个候选中偏向正确、排名高的商品 | GRPO、reward、KL、advantage | `rl.py`、`minionerec_trainer.py` | 5 |
| 离线评估 | 将生成候选转换为 HR/NDCG | Top-K、rank、merge | `evaluate.sh`、`evaluate.py`、`calc.py` | 6 |

## 4. 数据文件先看懂

仓库样例 `data/Amazon/index/` 中：

- `Industrial_and_Scientific.emb-qwen-td.npy`：商品文本向量，当前 NPY 头部显示 shape `(3686, 2560)`。
- `Industrial_and_Scientific.index.json`：`商品整数 id -> [<a_i>, <b_j>, <c_k>]`，默认每个商品 3 个 SID token。
- `Industrial_and_Scientific.item.json`：`商品整数 id -> 元数据`，至少有 `title`，通常还有 `description`。
- `data/Amazon/info/*.txt`：每行是 `完整 SID + 商品标题 + 原始 item_id`，用于约束解码和评估。
- `train/valid/test/*.csv`：每行是一个“用户历史 → 目标商品”样本。

CSV 关键字段：

```text
history_item_id    = [117, 118, ...]       # 原始商品 id
history_item_sid   = ['<a_...><b_...><c_...>', ...]
item_id            = 3681                  # 目标原始商品 id
item_sid           = '<a_...><b_...><c_...>'
history_item_title = ['商品标题 1', ...]
item_title         = '目标商品标题'
```

CSV 里的列表实际是字符串，`data.py` 用 `eval(...)` 把它恢复成 Python 列表。`eval` 对不可信输入有安全风险，生产代码应优先使用 `ast.literal_eval`；这里是项目原有实现。

## 5. 端到端课程大纲

### 第 1 课：入口和文件地图

**读完要能回答：** 主线从哪个 shell 启动，每个阶段由哪个 Python 文件负责？

阅读顺序：

1. `README.md` 的 Key Techniques、Repository Overview、Full Pipeline Walk-through。
2. `sft.sh`：看传给 `sft.py` 的模型、数据、词表参数。
3. `rl.sh`：看 `num_generations`、`beam_search`、`beta` 等 RL 参数。
4. `evaluate.sh`：看测试集切分、并行推理、合并和指标计算。
5. `requirements.txt`：知道 PyTorch、Transformers、TRL、Accelerate、DeepSpeed 的角色。

### 第 2 课：交互如何变成训练样本

**读完要能回答：** 一个用户的历史和目标商品如何写入 CSV，并如何变成 prompt？

主链：

```text
原始 Amazon reviews
  -> data/amazon18_data_process.py:process_dataset_recursive
  -> 时间和频次过滤
  -> 按时间排序、最多保留 10 个历史商品
  -> train/valid/test
  -> convert_dataset.py:convert_interactions_to_csv
  -> data.py:SidSFTDataset.pre
```

必读符号：

1. `data/amazon18_data_process.py:process_dataset_recursive`、`generate_interaction_list_json2csv_style`：看用户/商品过滤、时间排序和滑动窗口。`max(i - 10, 0)` 表示历史最多 10 项；`data/process.py:gao` 是较早的同类实现，可作为对照。
2. `convert_dataset.py:load_dataset`：读取 item、index 和 `.inter` 文件。
3. `convert_dataset.py:semantic_tokens_to_id`：把三个 token 拼为一个完整 SID 字符串。
4. `data.py:CSVBaseDataset`：读取 CSV，`self.data` 是 Pandas DataFrame，可理解为 `(样本数, 字段数)`。
5. `data.py:SidSFTDataset.get_history`：长度为 H 的历史 SID 列表 → 一个 prompt 字符串。

shape：

```text
一行 CSV（history 长度 H）
  -> input: str, output: str
  -> tokenizer
  -> input_ids: (L,), attention_mask: (L,), labels: (L,)
```

`L` 是 tokenizer 后的长度，不是历史商品数；一个 SID 字符串未必只对应 3 个语言模型 token。

### 第 3 课：文本向量和 SID

**读完要能回答：** 为什么一个商品会变成 `<a_...><b_...><c_...>`？

主链：

```text
title + description
  -> rq/text2emb/amazon_text2emb.py
  -> Transformer 编码器
  -> masked mean pooling
  -> .npy: (N, D=2560)
  -> rq/models/rqvae.py:RQVAE
  -> rq/models/rq.py:ResidualVectorQuantizer
  -> rq/models/vq.py:VectorQuantizer
  -> indices: (N, 3)
  -> rq/generate_indices.py
```

shape 变化（当前样例）：

1. 文本 tokenizer：`input_ids` 是 `(B, T)`，`T <= max_sent_len`，源码默认 2048。
2. 编码器：`last_hidden_state` 是 `(B, T, D)`。
3. `attention_mask.unsqueeze(-1)`：`(B,T)` → `(B,T,1)`，再 expand 为 `(B,T,D)`。
4. masked mean pooling：`sum_embeddings` 为 `(B,D)`，所有 batch 保存为 `(N,D)`；当前文件是 `(3686,2560)`。
5. RQ-VAE encoder：`(B,2560)` → `layers=[2048,1024,512,256,128,64]` 时依次变成 `(B,2048)`、`(B,1024)`、`(B,512)`、`(B,256)`、`(B,128)`、`(B,64)`，再变成 `(B,e_dim)`；当前 `rq/rqvae.py` 默认 `e_dim=32`，所以最后是 `(B,32)`。
6. 三个 VectorQuantizer 各返回与输入同形状的量化向量；索引 stack 后是 `(B,3)`。
7. decoder 把量化向量还原到 `(B,2560)`，用重建误差和量化误差训练。

3 层不是永远固定的，层数来自 `--num_emb_list 256 256 256` 的长度；改层数会同时改变 SID 长度和约束表。

必读：`rq/datasets.py:EmbDataset`、`rq/rqvae.py`、`rq/models/rqvae.py`、`rq/models/rq.py`、`rq/models/vq.py`、`rq/generate_indices.py`。

### 第 4 课：SFT 如何学习下一个 SID

**读完要能回答：** labels 为什么前面是 `-100`？模型究竟在哪些位置算 loss？

主链：

```text
sft.sh -> sft.py:train
  -> AutoModelForCausalLM / AutoTokenizer
  -> 从 index.json 收集并加入 SID token
  -> SidSFTDataset + SidItemFeatDataset + FusionSeqRecDataset
  -> ConcatDataset -> HuggingFace Dataset -> Trainer
```

`data.py:SidSFTDataset.pre` 的 shape：

```text
instruction: (I,)       prompt: (P,)       target SID: (Y,)
拼接: (I + P + Y,)
截断后 input_ids: (L,), attention_mask: (L,), labels: (L,)
batch padding 后: input_ids/attention_mask/labels = (B, L_batch)
模型 logits: (B, L_batch, V)
```

`labels = [-100] * input_prompt_len + target_tokens`：prompt 部分被交叉熵忽略，只有目标 SID 和 EOS 产生监督信号。

词表 shape：

```text
原 embedding.weight: (V_old, H)
add_tokens 后:       (V_old + V_sid, H)
```

`freeze_LLM=True` 时，`sft.py` 给旧词表行的梯度清零，只更新新 SID 行；打印的 trainable 参数数包含整个 embedding 矩阵，这是代码注释中已说明的统计口径。

必读符号：`sft.py:TokenExtender`、`sft.py:train`、`data.py:BaseDataset`、`data.py:SidSFTDataset.pre`、`data.py:SidItemFeatDataset`、`data.py:FusionSeqRecDataset`。

### 第 5 课：约束解码保证 SID 合法

**读完要能回答：** 模型每一步为什么不能在整个词表中任意选 token？

主链：

```text
info/*.txt 的完整 SID
  -> tokenizer
  -> hash_dict: 已生成前缀 -> 允许的下一个 token
  -> ConstrainedLogitsProcessor
  -> 非法 score 加 -inf
  -> beam search 只能选择合法 token
```

shape：

- 无 beam：`input_ids=(B,L)`，`scores=(B,V)`。
- `num_beams=K`：Transformers 展平为 `input_ids=(B*K,L)`、`scores=(B*K,V)`。
- processor 内部 view 为 `(-1,K,L)`，逐 batch、逐 beam 查询前缀。
- mask 与 scores 同形状 `(B*K,V)`；合法位置为 0，非法位置为 `-inf`。
- 生成结果 `sequences=(B*K,maxLen+newTokens)`，切后缀为 `(B*K,newTokens)`，再按 K 个候选分组。

`prefix_index=3`（非 GPT2）与 prompt 模板和 tokenizer 有关；换模型必须重新验证，不能只复制这个常数。

必读：`evaluate.py:main` 的 hash 表、`evaluate.py` 内部 `evaluate`、`LogitProcessor.py:ConstrainedLogitsProcessor.__call__`、`minionerec_trainer.py:prefix_allowed_tokens_fn/_prepare_inputs`。

### 第 6 课：RL/GRPO 如何使用推荐奖励

**读完要能回答：** 一条 prompt 如何生成 G 个候选，奖励如何进入 loss？

主链：

```text
rl.py:train
  -> SidDataset / RLTitle2SidDataset / RLSeqTitle2SidDataset
  -> Dataset.from_dict
  -> ReReTrainer
  -> 每个 prompt 重复 G 次并生成 completion
  -> rule/ranking/semantic/SASRec reward
  -> 组内 mean/std -> advantage
  -> 当前模型 log-prob + reference model KL -> GRPO loss
```

shape（B 是本地 prompt 数，G 是每 prompt 候选数，P/C 是 prompt/completion 长度）：

```text
prompt_ids:             (B*G,P)
prompt_mask:            (B*G,P)
completion_ids:         (B*G,C)
completion_mask:        (B*G,C)
拼接 input_ids:         (B*G,P+C)
模型 logits:            (B*G,C,V)
per_token_logps:        (B*G,C)
候选 reward:             (B*G,)
reward.view(-1,G):       (B,G)
advantage:               (B*G,)
```

奖励实现：

- `rule_reward`：文本与目标完全相等为 1，否则 0。
- `ndcg_rule_reward`：组内用排名相关奖励鼓励正确答案靠前。
- `semantic_reward`：商品 embedding cosine similarity；非法商品需要确认数据覆盖。
- `cf_reward`：调用 `sasrec.SASRec.forward_eval` 得到协同过滤分数。

`minionerec_trainer.py:compute_loss` 计算 `per_token_kl = exp(ref_logp-logp) - (ref_logp-logp) - 1`，再把 advantage 和 KL penalty 组合，最后用 `completion_mask` 忽略 EOS 后的 padding。第一次阅读先看默认分支，再看 `dapo`/`gspo`。

### 第 7 课：评估和指标

**读完要能回答：** 多 GPU 生成的 JSON 如何变成 HR@K/NDCG@K？

```text
evaluate.sh
  -> split.py 切分 test.csv
  -> evaluate.py 每个 shard 生成 JSON
  -> merge.py 合并
  -> calc.py 查 rank 并计算 HR@K/NDCG@K
```

shape：`EvalSidDataset(test=True)` 的每条 `input_ids=(L_i,)`；batch 左 padding 后 `input_ids=(B,maxLen)`；生成 `sequences=(B*num_beams,maxLen+newTokens)`；切片后 `(B*num_beams,newTokens)`；每 K 个候选组成一个样本的候选列表。

## 6. 推荐阅读表

| 主题 | 文件/符号 | 预计时间 | 读完能回答什么 |
| --- | --- | --- | --- |
| 项目地图 | `README.md`、`sft.sh`、`rl.sh`、`evaluate.sh` | 30 分钟 | 主流程从哪里启动？ |
| 数据清洗 | `data/amazon18_data_process.py:process_dataset_recursive`、`generate_interaction_list_json2csv_style` | 60 分钟 | 一条交互如何变成训练行？ |
| 文本 embedding | `rq/text2emb/amazon_text2emb.py:generate_item_embedding` | 45 分钟 | `(B,T,D)` 为什么变成 `(B,D)`？ |
| RQ-VAE | `rq/models/rqvae.py`、`rq/models/rq.py`、`rq/models/vq.py` | 90 分钟 | `(B,2560)` 如何变成 `(B,3)`？ |
| SID 文件 | `rq/generate_indices.py` | 30 分钟 | index.json 如何生成？ |
| 数据格式 | `convert_dataset.py` | 45 分钟 | item id 和 SID 如何对齐？ |
| SFT dataset | `data.py:SidSFTDataset`、`sft.py:train` | 90 分钟 | 哪些 token 参与 loss？ |
| 约束解码 | `evaluate.py`、`LogitProcessor.py` | 75 分钟 | 如何避免非法 SID？ |
| GRPO | `rl.py`、`minionerec_trainer.py` | 120 分钟 | reward 如何进入 loss？ |
| 结果评估 | `split.py`、`merge.py`、`calc.py` | 45 分钟 | 指标从哪里来？ |

## 7. 最小学习实验

### 实验 A：检查 CSV 和列表字符串

```bash
python - <<'PY'
import pandas as pd
path = 'data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv'
df = pd.read_csv(path)
print(df.shape)
print(df.iloc[0][['history_item_sid', 'item_sid']])
PY
```

### 实验 B：检查 embedding shape

```bash
python - <<'PY'
import numpy as np
x = np.load('data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy')
print(x.shape, x.dtype)
PY
```

预期当前样例为 `(3686, 2560)`，以本地实际输出为准。

### 实验 C：手算一个 SFT 样本

阅读 `data.py:SidSFTDataset.pre`，写出 `I`（instruction token 数）、`P`（prompt token 数）、`Y`（目标 token 数），验证 `L=min(I+P+Y,max_len)`，并说明为什么前 `I+P` 个 label 是 `-100`。

### 实验 D：手工理解约束解码

在 `LogitProcessor.py` 中构造一个只有两个允许下一个 token 的 `hash_dict`，观察非法位置为什么被加上 `-inf`。这比直接运行 50-beam 大模型更适合初学者。

## 8. 已补充中文头注释的关键文件

| 文件 | 第一阅读符号 | 顶部注释覆盖内容 |
| --- | --- | --- |
| `data.py` | `BaseDataset`、`SidSFTDataset` | Dataset 层级、SFT/RL 输出、shape |
| `sft.py` | `TokenExtender`、`train` | SFT 入口、词表扩展、batch shape |
| `rl.py` | `train`、reward 函数 | GRPO 入口、奖励、候选 shape |
| `minionerec_trainer.py` | `_prepare_inputs`、`compute_loss` | 采样、约束生成、GRPO loss |
| `LogitProcessor.py` | `ConstrainedLogitsProcessor.__call__` | logits mask 输入输出 shape |
| `evaluate.py` | `main`、内部 `evaluate` | 约束 beam search 和候选分组 |
| `convert_dataset.py` | `load_dataset`、`convert_interactions_to_csv` | item/SID/CSV 转换 |
| `rq/text2emb/amazon_text2emb.py` | `generate_item_embedding` | 文本到 `(N,D)` embedding |
| `rq/models/rqvae.py` | `forward`、`get_indices` | encoder → RQ → decoder |
| `rq/models/rq.py` | `forward` | 多层残差量化 |
| `rq/models/vq.py` | `forward` | codebook、索引和 loss shape |
| `rq/rqvae.py` | `parse_args`、`__main__` | RQ-VAE 训练启动 |
| `rq/datasets.py` | `EmbDataset` | NPY 到单条 Tensor |
| `rq/trainer.py` | `_train_epoch`、`_valid_epoch` | 重建 loss 和 collision rate |
| `data/amazon18_data_process.py` | `process_dataset_recursive`、`generate_interaction_list_json2csv_style` | 当前 Amazon18 过滤、滑窗和切分 |
| `data/process.py` | `gao` | 较早的 D3 风格过滤和滑窗实现 |
| `sasrec.py` | `SASRec.forward_eval` | RL 可选的协同过滤奖励 |
| `calc.py` | 主函数 | 从候选 JSON 计算 HR/NDCG |
| `split.py`、`merge.py` | `split`、`merge` | 多 GPU 评估文件切分与合并 |
| `config/zero2_opt.yaml` | DeepSpeed 配置项 | RL 分布式训练配置 |

## 9. 第一遍暂缓阅读的文件

- `sft_gpr.py`、`rl_gpr.py`：加入 GPR/VAFT/HEPO，先学会默认 SFT/RL。
- `ts_rec_data.py`、`ts_rec_sft.py`：新版 TS-Rec 支线，类名相似但 prompt 任务不同。
- `rq/rqkmeans_faiss.py`、`rq/rqkmeans_constrained.py`、`rq/rqkmeans_plus.py`：RQ-VAE 的替代 SID 算法。
- `utility.py`、`SASRecModules_ori.py`：传统推荐模型和旧工具代码。

## 10. 常见坑和源码边界

1. JSON 键一定是字符串；CSV 的 `item_id` 可能是整数，所以代码经常写 `str(item_id)`。
2. `'<a_1><b_2><c_3>'` 未必对应 tokenizer 的三个 token；真实长度以 tokenizer 输出为准。
3. `sft.py` 训练 tokenizer 使用 left padding；文本 embedding 脚本使用 right padding，不要混淆。
4. `num_generations` 必须整除全局 batch，`minionerec_trainer.py` 会检查。
5. `ConstrainedLogitsProcessor.count` 会随着生成步增加，每次新的 `generate` 都应创建新的 processor。
6. `evaluate.py` 把 `do_sample=False` 直接传给 `generate`，防止 checkpoint 配置把 beam search 静默变成采样。
7. 完整运行需要大 GPU；小环境先做 shape 检查和单元测试，不要直接运行默认 shell 脚本。
8. 不同 Transformers/TRL 版本可能影响 `logits_to_keep`、生成默认值和 DeepSpeed 行为；实际训练以版本和日志为准。
9. `rq/generate_indices.py` 当前把 dataset、checkpoint、输出目录写在文件级变量里，第一次运行前要先检查并修改这些路径；`rq/generate_indices_plus.py` 才提供较完整的命令行参数。

## 11. 学完后的复述题

1. `(2560,)` 商品文本向量为什么可以得到三个 SID token？
2. 历史长度 `H=10` 的样本，经过 tokenizer 后为什么不一定是 30 个 token？
3. SFT 的 `labels` 为什么把 prompt 部分设为 `-100`？
4. beam search 时 `scores` 为什么是 `(B*K,V)`？
5. GRPO 中 `reward.view(-1,G)` 和 `advantage` 分别表达什么？
6. `info/*.txt`、`index.json`、CSV 的 SID/item 信息如何对应？

如果答不出来，回到对应课程的“shape 变化”和“必读符号”重读；还不懂时可以继续让 AI 按“输入、每一步 shape、输出、一个样例”逐函数带读。

## 12. 事实边界

本导学只记录仓库能证明的事实：模型结构、数据流、训练入口、约束解码和指标计算。它没有证明个人职责，没有编造线上指标，也没有把论文结果当成本地复现实验结果。后续写简历或面试回答时，应补充真实职责、运行命令、日志和测量结果。
