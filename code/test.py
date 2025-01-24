import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import os
import pandas as pd


# def process_file(txt_file, region):
#     # 读取 txt 文件
#     df = pd.read_csv(txt_file, sep='\t', header=None)
#
#     # 删除第一列
#     df = df.drop(columns=0)
#     # print(df)
#
#     # 修改第三列，加上基因区域的前缀
#     df[2] = region + "_" + df[2].astype(str)
#
#     # 构造输出文件名
#     output_csv = txt_file.replace('.txt', '.csv')
#
#     print(df)
#
#     # 保存为 CSV 文件
#     df.to_csv(output_csv, index=False, header=False)
#     print(f"Processed and saved: {output_csv}")
#
#
# def process_all_files(directory):
#     # 遍历目录中的所有 txt 文件
#     for filename in os.listdir(directory):
#         if filename.endswith("_.txt"):
#             # 获取基因区域信息（例如从文件名中提取）
#             region = filename.split('_')[3]  # 假设格式是 feat_rank_new_基因区域.txt
#             txt_file = os.path.join(directory, filename)
#             process_file(txt_file, region)
#
#
# # 设定您的文件目录
# directory = "./feat_new"
#
# # 处理所有文件
# process_all_files(directory)




# def process_file(txt_file):
#     # 读取 txt 文件
#     df = pd.read_csv(txt_file, sep='\t', header=None)
#
#     # 删除第一列
#     df = df.drop(columns=0)
#
#     # 构造输出文件名
#     output_csv = txt_file.replace('.txt', '.csv')
#
#     # 保存为 CSV 文件
#     df.to_csv(output_csv, index=False, header=False)
#     print(f"Processed and saved: {output_csv}")
#
#
# # 设定您的文件路径
# txt_file = "./feat_new/feat_rank_new_ALLGeneALLRegion.txt"  # 请替换为实际的txt文件路径
#
# # 处理文件
# process_file(txt_file)

import os
import pandas as pd
from collections import Counter


def process_csv_files(file_paths):
    # 读取所有 csv 文件
    data = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, header=None, sep=',')
        print(df)
        data.append(df)

    # 假设每个文件有 3 列，先创建一个计数器来统计每列的所有值出现的次数
    count_column1 = Counter()
    count_column2 = Counter()
    count_column3 = Counter()

    # 遍历所有数据，统计每列的值出现的次数
    for df in data:
        count_column1.update(df[0])  # 第一列
        count_column2.update(df[1])  # 第二列
        count_column3.update(df[2])  # 第三列

    print(count_column1)
    print(count_column2)
    print(count_column3)

    mRNA = pd.DataFrame(sorted(count_column1.items(), key=lambda x: x[1], reverse=True), columns=['key', 'value'])
    methylation = pd.DataFrame(sorted(count_column2.items(), key=lambda x: x[1], reverse=True), columns=['key', 'value'])
    miRNA = pd.DataFrame(sorted(count_column3.items(), key=lambda x: x[1], reverse=True), columns=['key', 'value'])

    print(mRNA)
    print(methylation)
    print(miRNA)

    # mRNA.to_csv('F:\\run\\rank_mRNA.csv', index=False)
    methylation.to_csv('F:\\run\\rank_methylation.csv', index=False)
    # miRNA.to_csv('F:\\run\\rank_miRNA.csv', index=False)
    return df


# 指定8个 CSV 文件的路径
file_paths = [
    'F:\\run\\feat_new\\feat_rank_new_1stExon.csv', "F:\\run\\feat_new\\feat_rank_new_3'UTR.csv",
    "F:\\run\\feat_new\\feat_rank_new_5'UTR.csv", "F:\\run\\feat_new\\feat_rank_new_TSS200.csv",
    "F:\\run\\feat_new\\feat_rank_new_TSS1500.csv", "F:\\run\\feat_new\\feat_rank_new_Body.csv",
    "F:\\run\\feat_new\\feat_rank_new_ALLGeneALLRegion.csv", "F:\\run\\feat_new\\feat_rank_new_ALLRegion.csv"
]

# 处理并找出共同值
process_csv_files(file_paths)


#
# import os
# import pandas as pd


# def process_file(txt_file):
#     # 读取 txt 文件
#     df = pd.read_csv(txt_file, sep='\t', header=None)
#
#     # 删除第一列
#     df = df.drop(columns=0)
#
#     # 构造输出文件名
#     output_csv = txt_file.replace('.txt', '.csv')
#
#     # 保存为 CSV 文件
#     df.to_csv(output_csv, index=False, header=False)
#     print(f"Processed and saved: {output_csv}")
#
#
# def process_all_files(directory):
#     # 遍历目录中的所有 txt 文件
#     for filename in os.listdir(directory):
#         if filename.endswith(".txt"):
#             txt_file = os.path.join(directory, filename)
#             process_file(txt_file)
#
#
# # 设定您的文件目录
# directory = "./feat_new"  # 请替换为实际文件目录
#
# # 处理所有文件
# process_all_files(directory)

