# 阅读导航：
# 从参数看输入范围：dataset 选商品类目，st/ed 年月定义时间窗口，user_k 传给迭代过滤。
# 运行逻辑在 amazon18_data_process.py 末尾入口；输出目录包含 .inter、.item.json 和映射文件。
# 注意当前 Python 主线传给 k-core 的是 user_k，item_k 虽在脚本中提供但未参与该调用。
# Amazon18 数据预处理启动脚本：调用 amazon18_data_process.py 生成过滤后的交互数据。
# 先检查数据路径和 k-core 参数，再进入 Python 实现。
python amazon18_data_process.py \
    --dataset Industrial_and_Scientific \
    --user_k 5 \
    --item_k 5 \
    --st_year 1996 \
    --st_month 10 \
    --ed_year 2018 \
    --ed_month 10 \
    --output_path ./Amazon18
