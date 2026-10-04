"""RQ 训练公共工具。

本文件提供目录创建、日志颜色和时间格式化等小工具，供 rq/rqvae.py 与
rq/trainer.py 使用；它不改变模型结构，也不参与 SID 的计算。

先从 rq/trainer.py 的 import 找调用点：ensure_dir 建目录、
get_local_time 生成 checkpoint 时间名、set_color 格式化日志；
delete_file 只处理文件清理。这里没有模型计算。
"""

import datetime
import os


def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)

def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur

def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
