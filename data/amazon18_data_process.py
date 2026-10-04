"""Amazon18 原始评论与商品元数据的预处理入口（尚不生成 SID）。

第一次阅读从文件末尾 `if __name__ == '__main__'` 开始，沿真实调用顺序向上跳：
1. parse_args / get_timestamp_start：确定类目、输入文件和时间窗口。
2. load_reviews_json2csv_style：把 review JSONL 读成字典列表；此处不按时间过滤。
3. process_dataset_recursive：读取 metadata，验证标题，调用 k_core_filtering_json2csv_style；
   如果过滤后商品少于 3000 且起始年份大于 1996，就把起始年份向前扩一年重试。
4. convert_inters2dict_amazon18_style：为用户和商品编连续整数 id。
5. generate_interaction_list_json2csv_style：逐用户按时间排序；每个目标商品取之前
   最多 10 次交互作为历史，再把所有样本按目标时间排序。
6. convert_to_atomic_files_json2csv_style：按全局时间顺序 8:1:1 切分并写 .inter。
7. create_item_features_amazon18_style / load_review_data_amazon18_style：写商品和评论特征，
   最后写 .inter.json、.item.json、.review.json、.user2id、.item2id。

关键中间量：reviews 是原始评论列表；filtered_reviews 是过滤后的评论列表；
item2index 将原始 ASIN 映射到整数商品 id；interaction_list 的每条记录含历史 id
列表、目标 id、标题、评分和时间。下游 rq/text2emb 读取 .item.json；
convert_dataset.py 读取 .item.json、.index.json 和 .train/.valid/.test.inter。
阅读时先盯一条 review 的 reviewerID、asin、unixReviewTime 如何流到一条 .inter 行。
"""

import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import datetime
import torch
from tqdm import tqdm
import numpy as np


def clean_text(text):
    """清理标题或评论中的 HTML 标签、实体和多余空白，返回单行文本。"""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    # Decode HTML entities
    text = html.unescape(text)
    # Replace quotes
    text = text.replace("&quot;", "\"").replace("&amp;", "&")
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def check_path(path):
    """写输出文件前确保父目录存在。"""
    os.makedirs(path, exist_ok=True)


def write_json_file(data, file_path):
    """把字典或列表写为便于检查的缩进 JSON。"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def write_remap_index(index_map, file_path):
    """把原始用户/商品标识与连续整数 id 按制表符写到映射文件。"""
    with open(file_path, 'w') as f:
        for original, mapped in index_map.items():
            f.write(f"{original}\t{mapped}\n")


# Dataset name mapping
amazon18_dataset2fullname = {
    'Arts': 'Arts_Crafts_and_Sewing',
    'Games': 'Video_Games',
    'Sports': 'Sports_and_Outdoors',
    'Instruments': 'Musical_Instruments',
    'Scientific': 'Scientific'
}


def get_timestamp_start(year, month):
    """把指定年月的 1 日零点转换为 Unix 秒时间戳，供评论时间过滤。"""
    return int(datetime.datetime(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())


def load_metadata_json2csv_style(category, metadata_file=None):
    """读取商品 JSONL，并筛出标题可用的 ASIN。

    这里的 metadata 不是模型生成的文件，而是 Amazon 原始商品元数据文件。
    每行通常是一个 JSON 对象，至少包含 `asin` 和 `title`，还可能包含
    `description`、`brand`、`categories` 等字段。例如：
    `{"asin": "B001...", "title": "某个商品"}`。

    `metadata_file` 由命令行参数 `--metadata_file` 传入；如果不传，代码按
    当前工作目录计算默认路径 `../meta_{category}.json`。因此从仓库根目录
    直接运行时，它会找仓库上一级的文件；从 `data/` 目录运行时，才会找仓库
    根目录下的 `meta_{category}.json`。当前仓库没有这些原始 `meta_*.json`，
    只有已经处理好的 `data/Amazon/index/*.item.json`；后者是“整数 item_id →
    商品特征”的 JSON 对象，格式不同，不能直接当作这里的原始 metadata。

    函数返回 `(metadata, id_title, remove_items)`：`metadata` 是所有原始商品
    字典的列表；`id_title` 是本函数根据有效标题临时建立的
    `asin → 清洗后的 title` 字典；`remove_items` 是没有合格标题的 ASIN 集合。
    后续 k-core 用 `id_title` 判断评论中的 asin 是否可用，滑窗函数用它填充
    历史标题，最终 `create_item_features_amazon18_style` 再用完整 metadata
    生成下游的 `.item.json`。
    """
    # 注意：这是相对“启动 Python 时的当前目录”，不是相对本文件所在目录。
    if metadata_file is None:
        metadata_file = f'../meta_{category}.json'
    
    metadata = []
    try:
        with open(metadata_file) as f:
            metadata = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Metadata file {metadata_file} not found")
        return [], {}, set()
    
    # 这里才创建 id_title；调用方没有提前准备这个变量。
    id_title = {}
    remove_items = set()
    
    for meta in tqdm(metadata, desc="Processing metadata"):
        if ('title' not in meta) or (meta['title'].find('<span id') > -1):
            remove_items.add(meta['asin'])
            continue
        
        # Clean title like json2csv
        meta['title'] = meta["title"].replace("&quot;", "\"").replace("&amp;", "&").strip(" ").strip("\"")
        
        if len(meta['title']) > 1 and len(meta['title'].split(" ")) <= 20:
            id_title[meta['asin']] = meta['title']
            # Store full metadata for later use
        else:
            remove_items.add(meta['asin'])
    
    return metadata, id_title, remove_items


def load_reviews_json2csv_style(category, reviews_file=None, start_timestamp=None, end_timestamp=None):
    """读取 review JSONL 为列表；这里不按时间窗口过滤。

    时间过滤放在 k_core_filtering_json2csv_style 的迭代内，避免读入阶段
    提前改变参与用户/商品频次统计的样本集合。
    """
    if reviews_file is None:
        try:
            with open(f'../{category}_5.json') as f:
                reviews = [json.loads(line) for line in f]
        except FileNotFoundError:
            try:
                with open(f'../{category}.json') as f:
                    reviews = [json.loads(line) for line in f]
            except FileNotFoundError:
                print(f"Reviews file not found for category {category}")
                return []
    else:
        try:
            with open(reviews_file) as f:
                reviews = [json.loads(line) for line in f]
        except FileNotFoundError:
            print(f"Reviews file {reviews_file} not found")
            return []
    
    # NOTE: Don't filter by timestamp here, do it in k-core filtering like json2csv
    return reviews


def k_core_filtering_json2csv_style(reviews, id_title, K=5, start_timestamp=None, end_timestamp=None):
    """执行推荐数据常用的 K-core 过滤，保留活跃用户和热门商品。

    先理解三个字段：

    - `reviewerID`：Amazon 评论中的用户编号，可以理解为一个用户节点；
    - `asin`：Amazon Standard Identification Number，商品的唯一编号，可以理解为
      一个商品节点。它不是商品标题，也不是后面模型使用的整数 item_id；
    - 一条 review：一条“用户 reviewerID 在时间 unixReviewTime 与商品 asin 发生
      交互”的记录，同时可能包含评分和评论文本。

    K-core 的目标是让留下来的每个用户至少有 K 条有效交互、每个商品至少被
    K 条有效交互覆盖。代码先删除没有合格标题的 asin，再在时间窗口内统计用户
    和商品频次：频次小于 K 的用户/商品加入 remove_users/remove_items；下一轮
    删除与它们有关的 review 后重新统计。因为删除一个低频商品可能让某个用户
    也跌到 K 以下，所以必须循环到本轮没有新增删除对象为止。

    小例子（K=2）：A 评价 X、Y，B 评价 X，C 评价 X、Z。第一轮 Y、Z 各只有
    1 条，B 只有 1 条，会删除 Y、Z 和 B；第二轮 A、C 都只剩 X 这一条，
    又会删除 A、C；第三轮 X 也没有用户留下。最终可能没有数据，这正是
    K-core 过滤的级联效果，不是程序漏读。

    返回 `(new_reviews, user_counts, item_counts)`。注意当前实现只使用一个 K，
    调用处传入的是 `args.user_k`；命令行虽然声明了 `item_k`，但没有单独使用。
    """
    remove_users = set()
    remove_items = set()
    
    # 第一轮先删掉没有有效标题的商品；id_title 的键是原始 ASIN。
    for review in reviews:
        if review['asin'] not in id_title:
            remove_items.add(review['asin'])
    
    # k-core 必须迭代：删掉低频商品后，原本达标的用户也可能跌到 K 以下。
    while True:
        new_reviews = []
        flag = False
        total = 0
        user_counts = dict()
        item_counts = dict()
        
        for review in tqdm(reviews, desc="K-core filtering"):
            # 时间过滤放在每轮统计之前：只有窗口内的 review 才计入频次。
            if start_timestamp and end_timestamp:
                if int(review["unixReviewTime"]) < start_timestamp or int(review["unixReviewTime"]) > end_timestamp:
                    continue
            
            # 只要用户或商品已被上一轮淘汰，这条“用户-商品边”也一起淘汰。
            if review['reviewerID'] in remove_users or review['asin'] in remove_items:
                continue
            
            if review['reviewerID'] not in user_counts:
                user_counts[review['reviewerID']] = 0
            user_counts[review['reviewerID']] += 1
            
            if review['asin'] not in item_counts:
                item_counts[review['asin']] = 0
            item_counts[review['asin']] += 1
            
            total += 1
            new_reviews.append(review)
        
        # 本轮频次低于 K 的用户/商品加入永久删除集合，下一轮重新统计。
        # user_counts 的 key 是 reviewerID，item_counts 的 key 是 ASIN。
        for user in user_counts:
            if user_counts[user] < K:
                remove_users.add(user)
                flag = True
        
        for item in item_counts:
            if item_counts[item] < K:
                remove_items.add(item)
                flag = True
        
        print(f"Users: {len(user_counts)}, Items: {len(item_counts)}, Reviews: {total}, Density: {total / (len(user_counts) * len(item_counts)) if len(user_counts) > 0 and len(item_counts) > 0 else 0}")
        
        if not flag:
            break
        
        reviews = new_reviews
    
    return new_reviews, user_counts, item_counts


def convert_inters2dict_amazon18_style(reviews):
    """按用户时间排序，同时建立原始用户/ASIN 到连续整数 id 的映射。

    返回 user2items、user2index、item2index 和 interactions；整数商品 id
    供 .inter、.item.json 和后续 SID 索引共同使用。这里不生成训练样本窗口。
    """
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    
    # Sort reviews by timestamp for each user
    user_reviews = collections.defaultdict(list)
    for review in reviews:
        user_reviews[review['reviewerID']].append(review)
    
    # Sort each user's reviews by timestamp (don't remove duplicates like json2csv)
    for user in user_reviews:
        user_reviews[user].sort(key=lambda x: int(x['unixReviewTime']))
    
    # Create mappings and interactions
    interactions = []
    for user in user_reviews:
        if user not in user2index:
            user2index[user] = len(user2index)
        
        user_items = []
        for review in user_reviews[user]:
            item = review['asin']
            if item not in item2index:
                item2index[item] = len(item2index)
            
            user_items.append(item)
            interactions.append((
                user, item, 
                float(review['overall']), 
                int(review['unixReviewTime'])
            ))
        
        user2items[user2index[user]] = [item2index[item] for item in user_items]
    
    return user2items, user2index, item2index, interactions


def generate_interaction_list_json2csv_style(reviews, user2index, item2index, id_title):
    """为每个用户的第 2 次及以后交互构造“历史→当前目标”样本。

    同一用户先按 unixReviewTime 排序；第 i 次交互使用 [max(i-10,0):i]
    作为历史，所以历史最多 10 项。最后按目标时间全局排序，交给下一函数切分。
    返回的每项按位置保存用户、历史 ASIN、目标 ASIN、历史/目标整数 id、
    标题、评分和时间；读下面 append 的 11 个位置时可逐项对照。
    """
    # Create user interactions similar to json2csv
    interact = dict()
    item2id = {item: idx for item, idx in item2index.items()}
    
    for review in tqdm(reviews, desc="Building interaction list"):
        user = review['reviewerID']
        item = review['asin']
        
        if user not in interact:
            interact[user] = {
                'items': [],
                'ratings': [],
                'timestamps': [],
                'item_ids': [],
                'titles': []
            }
        
        # Keep all interactions like json2csv (no deduplication)
        interact[user]['items'].append(item)
        interact[user]['ratings'].append(review['overall'])
        interact[user]['timestamps'].append(review['unixReviewTime'])
        interact[user]['item_ids'].append(item2id[item])
        interact[user]['titles'].append(id_title[item])
    
    # Sort by timestamp for each user
    interaction_list = []
    for user in tqdm(interact.keys(), desc="Creating interaction sequences"):
        items = interact[user]['items']
        ratings = interact[user]['ratings']
        timestamps = interact[user]['timestamps']
        item_ids = interact[user]['item_ids']
        titles = interact[user]['titles']
        
        # Sort all by timestamp
        all_data = list(zip(items, ratings, timestamps, item_ids, titles))
        all_data.sort(key=lambda x: int(x[2]))
        items, ratings, timestamps, item_ids, titles = zip(*all_data)
        items, ratings, timestamps, item_ids, titles = list(items), list(ratings), list(timestamps), list(item_ids), list(titles)
        
        # Create sequences like json2csv (sliding window with max history of 10)
        for i in range(1, len(items)):
            st = max(i - 10, 0)
            interaction_list.append([
                user,                    # 原始用户 id
                items[st:i],            # 历史商品 ASIN 列表
                items[i],               # 当前目标商品 ASIN
                item_ids[st:i],         # 历史商品整数 id 列表
                item_ids[i],            # 目标商品整数 id
                titles[st:i],           # 历史商品标题列表
                titles[i],              # 目标商品标题
                ratings[st:i],          # 历史评分列表
                ratings[i],             # 目标评分
                timestamps[st:i],       # 历史时间戳列表
                timestamps[i]           # 目标时间戳
            ])
    
    # 先把所有用户的样本合并并按目标时间排序，下一函数才按 8:1:1 切分。
    interaction_list.sort(key=lambda x: int(x[-1]))
    return interaction_list


def convert_to_atomic_files_json2csv_style(args, interaction_list, user2index):
    """把已按目标时间排序的样本按 80%/10%/10% 切为 .inter 文件。

    每行是 `user_id:token`、空格分隔的历史 `item_id_list:token_seq`、
    目标 `item_id:token`。写入的是整数 id，不是 SID；SID 在后续转换步骤加入。
    """
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    
    # Create output directories
    check_path(os.path.join(args.output_path, args.dataset))
    
    # 切分单位是滑窗样本，不是用户；同一用户的样本可能跨 train/valid/test。
    total_len = len(interaction_list)
    train_end = int(total_len * 0.8)
    valid_end = int(total_len * 0.9)
    
    train_interactions = interaction_list[:train_end]
    valid_interactions = interaction_list[train_end:valid_end]
    test_interactions = interaction_list[valid_end:]
    
    print(f"Train interactions: {len(train_interactions)}")
    print(f"Valid interactions: {len(valid_interactions)}")
    print(f"Test interactions: {len(test_interactions)}")
    
    # Write train file
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for interaction in train_interactions:
            user_id_original = interaction[0]
            user_id = user2index[user_id_original]  
            history_item_ids = [str(x) for x in interaction[3]]  # history item ids
            target_item_id = str(interaction[4])  # target item id
            
            # Limit history to last 50 items like amazon18
            history_seq = history_item_ids[-50:]
            file.write(f'{user_id}\t{" ".join(history_seq)}\t{target_item_id}\n')
    
    # Write valid file  
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for interaction in valid_interactions:
            user_id_original = interaction[0]
            user_id = user2index[user_id_original]  
            history_item_ids = [str(x) for x in interaction[3]]  # history item ids
            target_item_id = str(interaction[4])  # target item id
            
            # Limit history to last 50 items like amazon18
            history_seq = history_item_ids[-50:]
            file.write(f'{user_id}\t{" ".join(history_seq)}\t{target_item_id}\n')
    
    # Write test file
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for interaction in test_interactions:
            user_id_original = interaction[0]
            user_id = user2index[user_id_original] 
            history_item_ids = [str(x) for x in interaction[3]]  # history item ids  
            target_item_id = str(interaction[4])  # target item id
            
            # Limit history to last 50 items like amazon18
            history_seq = history_item_ids[-50:]
            file.write(f'{user_id}\t{" ".join(history_seq)}\t{target_item_id}\n')
    
    return train_interactions, valid_interactions, test_interactions


def load_review_data_amazon18_style(reviews, user2index, item2index):
    """为保留的评论建立 (用户 id, 商品 id, 时间) 到文本特征的映射。

    reviewText 和 summary 经 clean_text 清洗，写入 .review.json；它与 .inter
    训练样本不同，不参与这里的时间切分。
    """
    review_data = {}
    
    for review in tqdm(reviews, desc='Load reviews'):
        try:
            user = review['reviewerID']
            item = review['asin']
            
            if user in user2index and item in item2index:
                uid = user2index[user]
                iid = item2index[item]
                
                # Use timestamp to create unique keys for same user-item pairs at different times
                timestamp = review['unixReviewTime']
                unique_key = str((uid, iid, timestamp))
                
            else:
                continue
                
            if 'reviewText' in review:
                review_text = clean_text(review['reviewText'])
            else:
                review_text = ''
                
            if 'summary' in review:
                summary = clean_text(review['summary'])
            else:
                summary = ''
                
            review_data[unique_key] = {"review": review_text, "summary": summary}
            
        except (ValueError, KeyError):
            continue
    
    return review_data


def create_item_features_amazon18_style(metadata, item2index, id_title):
    """只为过滤后保留的商品生成 .item.json 特征。

    通过 item2index 对齐整数商品 id；从 metadata 提取标题、描述、品牌和类别。
    下游文本向量脚本主要读取 title 与 description。
    """
    item2feature = collections.defaultdict(dict)
    
    # Create a mapping from asin to metadata
    asin_to_meta = {}
    for meta in metadata:
        asin_to_meta[meta['asin']] = meta
    
    for item_asin, item_id in item2index.items():
        if item_asin in asin_to_meta:
            meta = asin_to_meta[item_asin]
            
            title = id_title.get(item_asin, clean_text(meta.get("title", "")))
            
            descriptions = meta.get("description", "")
            if descriptions:
                descriptions = clean_text(descriptions)
            else:
                descriptions = ""
                
            brand = meta.get("brand", "").replace("by\n", "").strip()
            
            categories = meta.get("categories", [])
            if categories and len(categories) > 0:
                # Handle both list and string formats
                if isinstance(categories[0], list):
                    # Flatten nested categories
                    flat_categories = []
                    for cat_group in categories:
                        flat_categories.extend(cat_group)
                    categories = flat_categories
                
                # Filter out invalid categories
                new_categories = []
                for category in categories:
                    if "</span>" not in str(category):
                        new_categories.append(str(category).strip())
                categories = ",".join(new_categories).strip()
            else:
                categories = ""
            
            item2feature[item_id] = {
                "title": title,
                "description": descriptions,
                "brand": brand,
                "categories": categories
            }
    
    return item2feature


def process_dataset_recursive(args, metadata, reviews, start_timestamp, end_timestamp):
    """读取元数据并进行时间窗内的迭代 k-core 过滤。

    若保留商品少于 3000 且起始年份仍大于 1996，修改 args.st_year 并递归重试。
    参数 metadata 在当前实现中只是历史遗留参数，调用时传入 None；函数内部
    重新调用 load_metadata_json2csv_style，并从 args.metadata_file 读取真正的
    外部文件。只返回过滤结果和 metadata；映射、滑窗、切分在文件末尾入口继续执行。
    """
    
    # Load metadata 
    metadata, id_title, remove_items = load_metadata_json2csv_style(
        args.dataset, args.metadata_file
    )
    
    if not metadata:
        print(f"Error: No metadata found for dataset {args.dataset}")
        return None
    
    print(f"Loaded {len(metadata)} metadata items, {len(id_title)} with valid titles")
    
    # Perform k-core filtering with timestamp filtering inside
    print("Performing k-core filtering...")
    filtered_reviews, user_counts, item_counts = k_core_filtering_json2csv_style(
        reviews, id_title, args.user_k, start_timestamp, end_timestamp
    )
    
    print(f"After filtering: {len(user_counts)} users, {len(item_counts)} items, {len(filtered_reviews)} reviews")
    
    # Check if we need to expand time range (like json2csv)
    if args.st_year > 1996 and len(item_counts) < 3000:
        print(f"Items count {len(item_counts)} < 3000, expanding time range...")
        args.st_year -= 1
        new_start_timestamp = get_timestamp_start(args.st_year, args.st_month)
        print(f"New time range: {args.st_year}-{args.st_month} to {args.ed_year}-{args.ed_month}")
        return process_dataset_recursive(args, metadata, reviews, new_start_timestamp, end_timestamp)
    
    return filtered_reviews, user_counts, item_counts, metadata, id_title


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Arts', help='Instruments / Arts / Games / Sports')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--st_year', type=int, default=1996, help='start year')
    parser.add_argument('--st_month', type=int, default=10, help='start month')
    parser.add_argument('--ed_year', type=int, default=2018, help='end year')
    parser.add_argument('--ed_month', type=int, default=11, help='end month')
    parser.add_argument('--metadata_file', type=str, default=None, help='metadata file path')
    parser.add_argument('--reviews_file', type=str, default=None, help='reviews file path')
    parser.add_argument('--output_path', type=str, default='./data', help='output directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print(f'Processing dataset: {args.dataset}')
    print(f'Initial time range: {args.st_year}-{args.st_month} to {args.ed_year}-{args.ed_month}')
    print(f'K-core threshold: {args.user_k}')
    
    # Set time range
    start_timestamp = get_timestamp_start(args.st_year, args.st_month)
    end_timestamp = get_timestamp_start(args.ed_year, args.ed_month)
    
    # Load reviews first (without timestamp filtering)
    print("Loading reviews...")
    reviews = load_reviews_json2csv_style(
        args.dataset, args.reviews_file
    )
    
    if not reviews:
        print(f"Error: No reviews found for dataset {args.dataset}")
        print("Please check if the reviews file exists and try again.")
        exit(1)
        
    print(f"Loaded {len(reviews)} total reviews")
    
    # Process dataset with recursive logic
    result = process_dataset_recursive(args, None, reviews, start_timestamp, end_timestamp)
    
    if result is None:
        print("Failed to process dataset")
        exit(1)
    
    filtered_reviews, user_counts, item_counts, metadata, id_title = result
    
    print(f"Final filtering results:")
    print(f"Users: {len(user_counts)}, Items: {len(item_counts)}, Reviews: {len(filtered_reviews)}")
    print(f"Density: {len(filtered_reviews) / (len(user_counts) * len(item_counts)) if len(user_counts) > 0 and len(item_counts) > 0 else 0}")
    
    # Convert to amazon18 style format
    print("Converting to amazon18 format...")
    user2items, user2index, item2index, interactions = convert_inters2dict_amazon18_style(filtered_reviews)
    
    print(f"After amazon18 conversion:")
    print(f"  User2index: {len(user2index)} users")
    print(f"  Item2index: {len(item2index)} items") 
    print(f"  Interactions: {len(interactions)} interactions")
    
    # Generate interaction list for 8:1:1 split like json2csv
    print("Generating interaction list for 8:1:1 split...")
    interaction_list = generate_interaction_list_json2csv_style(
        filtered_reviews, user2index, item2index, id_title
    )
    
    print(f"Generated {len(interaction_list)} interaction sequences")
    
    # Create output directory and split data
    train_interactions, valid_interactions, test_interactions = convert_to_atomic_files_json2csv_style(
        args, interaction_list, user2index
    )
    
    # 用户→商品序列另存 JSON，供其他读取方式使用；它不替代上面的 .inter。
    user2items_final = collections.defaultdict(list)
    for user_idx, item_list in user2items.items():
        user2items_final[user_idx] = item_list
    
    # Write interaction files (amazon18 style output)
    write_json_file(user2items_final, os.path.join(args.output_path, args.dataset, f'{args.dataset}.inter.json'))
    
    # Create item features
    print("Creating item features...")
    item2feature = create_item_features_amazon18_style(metadata, item2index, id_title)
    
    # Load review data
    print("Loading review data...")
    review_data = load_review_data_amazon18_style(filtered_reviews, user2index, item2index)
    
    print(f"Final statistics:")
    print(f"Users: {len(user2index)}")
    print(f"Items: {len(item2index)}")
    print(f"Reviews: {len(review_data)}")
    print(f"Total interaction sequences: {len(interaction_list)}")
    print(f"Train sequences: {len(train_interactions)}")
    print(f"Valid sequences: {len(valid_interactions)}")
    print(f"Test sequences: {len(test_interactions)}")
    
    # Write output files (amazon18 style)
    write_json_file(item2feature, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item.json'))
    write_json_file(review_data, os.path.join(args.output_path, args.dataset, f'{args.dataset}.review.json'))
    
    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2id'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2id'))
    
    print("Processing completed!")
