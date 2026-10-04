# 阅读导航：
# 阅读顺序：模型路径和类别 → split.py 的 GPU 分片 → evaluate.py 的约束 beam search → merge.py → calc.py。
# 每个 GPU 生成编号.json；合并后的 final_result_类别.json 含真实目标和 predict 候选，calc.py 再算指标。
# --num_beams 既影响搜索宽度，也影响每条样本返回的候选数；这里的 GPU 列表要与 split 参数一致。
# MiniOneRec 离线评估脚本：split -> evaluate -> merge -> calc。
# 评估类别：可改为 Office_Products，或在这里填写多个类别。
for category in "Industrial_and_Scientific"
do
    # 待评估的 SFT/RL 模型目录；这里必须替换为实际 checkpoint 路径。
    exp_name="xxx"

    exp_name_clean=$(basename "$exp_name")
    echo "Processing category: $category with model: $exp_name_clean (STANDARD MODE)"
    
    # 训练文件只用于保留目录约定；当前评估主链实际读取 test_file。
    train_file=$(ls ./data/Amazon/train/${category}*.csv 2>/dev/null | head -1)
    # 测试集和 info 文件分别提供待预测样本、合法 SID 与商品映射。
    test_file=$(ls ./data/Amazon/test/${category}*11.csv 2>/dev/null | head -1)
    info_file=$(ls ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1)
    
    if [[ ! -f "$test_file" ]]; then
        echo "Error: Test file not found for category $category"
        continue
    fi
    if [[ ! -f "$info_file" ]]; then
        echo "Error: Info file not found for category $category"
        continue
    fi
    
    # 临时目录保存切分后的 CSV 和各 GPU 的 JSON 结果。
    temp_dir="./temp/${category}-${exp_name_clean}"
    echo "Creating temp directory: $temp_dir"
    mkdir -p "$temp_dir"
    
    echo "Splitting test data..."
    # 按 GPU 列表切分测试集；每个 shard 后续由一张 GPU 独立生成。
    python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "0,1,2,3,4,5,6,7"
    
    if [[ ! -f "$temp_dir/0.csv" ]]; then
        echo "Error: Data splitting failed for category $category"
        continue
    fi
    
    # 这里的编号必须和 split.py 使用的 cuda_list 保持一致。
    cudalist="0 1 2 3 4 5 6 7"
    echo "Starting parallel evaluation (STANDARD MODE)..."
    for i in ${cudalist}
    do
        if [[ -f "$temp_dir/${i}.csv" ]]; then
            echo "Starting evaluation on GPU $i for category ${category}"
            # 每个进程只看到一张物理 GPU；evaluate.py 内部执行约束 beam search。
            CUDA_VISIBLE_DEVICES=$i python -u ./evaluate.py \
                --base_model "$exp_name" \
                --info_file "$info_file" \
                --category ${category} \
                --test_data_path "$temp_dir/${i}.csv" \
                --result_json_data "$temp_dir/${i}.json" \
                --batch_size 8 \
                --num_beams 50 \
                --max_new_tokens 256 \
                --length_penalty 0.0 &  # 进程放到后台，随后由 wait 统一等待
        else
            echo "Warning: Split file $temp_dir/${i}.csv not found, skipping GPU $i"
        fi
    done
    echo "Waiting for all evaluation processes to complete..."
    wait
    
    # 所有后台进程结束后，检查是否至少生成了一个 shard 结果。
    result_files=$(ls "$temp_dir"/*.json 2>/dev/null | wc -l)
    if [[ $result_files -eq 0 ]]; then
        echo "Error: No result files generated for category $category"
        continue
    fi
    
    output_dir="./results/${exp_name_clean}"
    echo "Creating output directory: $output_dir"
    mkdir -p "$output_dir"

    # 从实际生成的 JSON 文件名恢复 shard 编号，避免合并不存在的结果。
    actual_cuda_list=$(ls "$temp_dir"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')
    echo "Merging results from GPUs: $actual_cuda_list"
    
    # 按 shard 编号合并候选结果；此步骤只拼接数据，不计算指标。
    python ./merge.py \
        --input_path "$temp_dir" \
        --output_path "$output_dir/final_result_${category}.json" \
        --cuda_list "$actual_cuda_list"
    
    if [[ ! -f "$output_dir/final_result_${category}.json" ]]; then
        echo "Error: Result merging failed for category $category"
        continue
    fi
    
    # calc.py 根据真实 item_sid 在候选列表中的排名计算 HR@K/NDCG@K。
    echo "Calculating metrics..."
    python ./calc.py \
        --path "$output_dir/final_result_${category}.json" \
        --item_path "$info_file"
    
    echo "Completed processing for category: $category"
    echo "Results saved to: $output_dir/final_result_${category}.json"
    echo "----------------------------------------" 
done

echo "All categories processed!"
