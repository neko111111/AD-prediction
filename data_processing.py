import numpy as np
import pandas as pd
from pathlib import Path
from model import FullConnect, Self_Attention, VCDN
import model
import torch
from sklearn.utils import shuffle
import utils
from sklearn.feature_selection import VarianceThreshold
from utils import filter_mean_var, anova, normalize
from sklearn import preprocessing
from train_test import prepare_trte_data

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

path1 = Path('D:/graduate/GPL13534')
path2 = Path('D:/graduate/ROSMAP-DATA20221011')
path3 = Path('D:/graduate/ROSMAP_Meta')
path4 = Path('D:/graduate/data')
path5 = Path('D:/graduate/ROSMAP_2')
path6 = Path('D:/graduate/ROSMAP_3')


# data1 = data[data.cogdx.isin([4.0, 5.0])]
# data2 = data[data.cogdx.isin([1.0, 2.0, 3.0, 6.0])]
# data2['is_AD'] = pd.Series()
# data2.drop(['cogdx'], axis=1, inplace=True)
# for i in range(len(data2)):
#     data2['is_AD'].iloc[i] = 0
# print(data2)
# data2.to_csv(path2/'not_AD_sample.csv', sep='\t', index=0)
# a = pd.concat([data['individualID'], data['specimenID'], data['organ'], data['tissue'], data['BrodmannArea']], axis=1)

# data = pd.read_csv(path/'ROSMAP_arraymiRNA.gct', sep='\t', low_memory=False, encoding='utf-8')
# data.to_csv(path/'ROSMAP_arraymiRNA.csv', sep='\t', index=False)


# a = pd.concat([data['ID'], data['Probe_SNPs'], data['UCSC_RefGene_Name'], data['UCSC_RefGene_Accession'],
#                data['UCSC_RefGene_Group'], data['Relation_to_UCSC_CpG_Island']], axis=1)
#
# a.to_csv(path1/'GPL13534-11288-simple.csv', sep=',', index=False)

# with open(path2/'AMP-AD_ROSMAP_Rush-Broad_IlluminaHumanMethylation450_740_imputed.tsv') as file1:
#     a = file1.readline()
#
# a = a.split('\t')[1:]
# a[-1] = "PT-BZD3"
#
#
# file2 = pd.read_csv(path3/'ROSMAP_biospecimen_metadata.csv', sep=',', encoding='utf-8')
# A = file2["individualID"].tolist()
# B = file2["specimenID"].tolist()
#
# for i, v in enumerate(a):
#     if v in B:
#         pos = B.index(v)
#         a[i] = A[pos]
#
# a.insert(0, "TargetID")
#
# print(a)


# 平均值方差过滤
def mean_var(file_name, feature_name, mean, var):
    data1 = pd.read_csv(path4 / file_name, sep=',', index_col=None, low_memory=False)
    print(data1)
    data1.drop(columns=[feature_name, 'disease_type'], axis=1, inplace=True)
    data1 = filter_mean_var(data1, mean, var)
    columns = pd.DataFrame(data1.columns)
    print(data1)
    print(columns)
    columns.to_csv(path5 / '3_featname.csv', sep=',', index=False, header=False)
    data1 = normalize(data1)  # 归一化
    print(data1)
    data1.to_csv(path5 / '3_tr.csv', sep=',', index=False, header=False)


# ANOVA分析过滤
def ANOVA(file_name, feature_name):
    a = pd.read_csv(path4 / file_name, sep=',', index_col=None, low_memory=False)
    sample = pd.Series(a[feature_name])
    a.drop(columns=[feature_name], inplace=True)
    print(a)
    feature = list(a.columns)
    columns = a.columns[:-1]
    n = len(columns)
    name = ["h" + str(i) for i in range(n)]
    name.append('disease_type')
    a.columns = name
    features = {}
    for i in range(n):
        features[name[i]] = feature[i]
    res = []
    for column in name:
        table = anova(a, column, 'disease_type')
        if table['PR(>F)'].iloc[0] < 0.05:
            res.append(column)
    a = a[res]
    a.insert(0, feature_name, sample)
    column = list(a.columns[1:-1])
    names = ["TargetID"]
    for c in column:
        names.append(features[c])
    names.append('disease_type')
    a.columns = names
    print(a)
    a.to_csv(path4 / 'mRNA_ANOVA.csv', sep=',', index=None)

# 主体begin
# 挑选样本个数
# data = pd.read_csv(path4/'disease_type.csv', sep='\t', low_memory=False)
# data.rename(columns={'sample': 'individualID', 'disease_type': 'cogdx'}, inplace=True)
# a = []
# dt = data.cogdx.tolist()
# for d in dt:
#     if d == 1.0:
#         a.append(0.0)
#     elif d == 4.0 or d == 5.0:
#         a.append(1.0)
#     else:
#         a.append(2.0)
# data['disease_type'] = a
# data = data[data['disease_type'] != 2.0]
# data.rename(columns={'individualID': 'sample'}, inplace=True)
# data.to_csv(path4/'sample_disease_type.csv', index=False, sep=',')


# 二分类
# data = pd.read_csv(path4/'final_miRNA.csv', sep='\t', low_memory=False)
# print(data)
# sample = pd.read_csv(path4/'sample_disease_type.csv', sep=',', low_memory=False)
# print(sample)
# # data.rename(columns={'Genes': 'gene_id'}, inplace=True)
# sample.rename(columns={'sample': 'gene_id'}, inplace=True)
# data = pd.merge(sample, data, on='gene_id')
# print(data)
# data.drop(columns=['cogdx'], inplace=True)
# data.insert(310, 'disease_type', data.pop('disease_type'))
# # data.rename(columns={'gene_id': 'Genes'}, inplace=True)
# print(data)
# data.to_csv(path4/'final_miRNA_2.csv', index=False, sep=',')


# 三分类
# data1 = pd.read_csv(path4 / 'final_DNAmethylation_2.csv', sep='\t', index_col=None, low_memory=False)
# dt = pd.read_csv(path4 / 'sample_disease_type_3.csv', sep=',', index_col=None, low_memory=False)
# dt.rename(columns={'sample': 'TargetID'}, inplace=True)
# print(data1)
# print(dt)
# data1 = pd.merge(dt, data1, on='TargetID')
# data1.drop(columns=['cogdx_x', 'cogdx_y', 'is_AD'], inplace=True)
# data1.insert(23789, 'disease_type_3', data1.pop('disease_type_3'))
# print(data1)
# dt.to_csv(path4/'sample_disease_type_3.csv', index=False, sep=',')
# data1.to_csv(path4/'final_DNAmethylation_3.csv', index=False, sep=',')


# 从样本中取出351个样本，二分类正样本182个，负样本169个；三分类各117个
# data = pd.read_csv(path4/'sample_disease_type.csv', sep=',', low_memory=False)
# data.rename(columns={'sample': 'individualID'}, inplace=True)
# print(data)
# cnt1 = 0
# cnt2 = 0
# # cnt3 = 0
# positive = []
# negative = []
# # mid = []
# for i in range(376):
#     if data.disease_type.iloc[i] == 0.0:
#         positive.append([data.individualID.iloc[i], data.cogdx.iloc[i], data.disease_type.iloc[i]])
#         cnt1 += 1
#     if data.disease_type.iloc[i] == 1.0 and cnt2 < 182:
#         negative.append([data.individualID.iloc[i], data.cogdx.iloc[i], data.disease_type.iloc[i]])
#         cnt2 += 1
#     # if data.disease_type_3.iloc[i] == 2.0 and cnt3 < 117:
#     #     mid.append([data.individualID.iloc[i], data.cogdx.iloc[i], data.disease_type.iloc[i]])
#     #     cnt3 += 1
# positive = pd.DataFrame(positive)
# negative = pd.DataFrame(negative)
# # mid = pd.DataFrame(mid)
# positive.rename(columns={0: 'sample', 1: 'cogdx', 2: 'disease_type'}, inplace=True)
# negative.rename(columns={0: 'sample', 1: 'cogdx', 2: 'disease_type'}, inplace=True)
# # mid.rename(columns={0: 'sample', 1: 'cogdx', 2: 'disease_type_3'}, inplace=True)
# sample = pd.concat((positive, negative)).reset_index()
# sample.drop(columns=['index'], axis=1, inplace=True)
# sample = shuffle(sample)
# print(sample)
# sample.to_csv(path4/'sample_disease_type_351samples.csv', index=False, sep=',')


# 训练集的划分
# data = pd.read_csv(path4/'sample_disease_type_351samples.csv', sep=',', low_memory=False)
# data.rename(columns={'sample': 'individualID'}, inplace=True)
# print(data)
# cnt1 = 0
# cnt2 = 0
# # cnt3 = 0
# positive = []
# negative = []
# # mid = []
# for i in range(351):
#     if data.disease_type.iloc[i] == 0.0 and cnt1 < 123:
#         positive.append([data.individualID.iloc[i], data.cogdx.iloc[i], data.disease_type.iloc[i]])
#         cnt1 += 1
#     if data.disease_type.iloc[i] == 1.0 and cnt2 < 122:
#         negative.append([data.individualID.iloc[i], data.cogdx.iloc[i], data.disease_type.iloc[i]])
#         cnt2 += 1
#     # if data.disease_type_3.iloc[i] == 2.0 and cnt3 < 82:
#     #     mid.append([data.individualID.iloc[i], data.cogdx.iloc[i], data.disease_type_3.iloc[i]])
#     #     cnt3 += 1
# positive = pd.DataFrame(positive)
# negative = pd.DataFrame(negative)
# # mid = pd.DataFrame(mid)
# positive.rename(columns={0: 'sample', 1: 'cogdx', 2: 'disease_type'}, inplace=True)
# negative.rename(columns={0: 'sample', 1: 'cogdx', 2: 'disease_type'}, inplace=True)
# # mid.rename(columns={0: 'sample', 1: 'cogdx', 2: 'disease_type_3'}, inplace=True)
# sample = pd.concat((positive, negative)).reset_index()
# sample.drop(columns=['index'], axis=1, inplace=True)
# sample = shuffle(sample)
# print(sample)
# sample.to_csv(path4/'sample_tr.csv', index=False, sep=',')


# 测试集的划分
# data = pd.read_csv(path4/'sample_disease_type_351samples.csv', sep=',', low_memory=False)
# sample_tr = pd.read_csv(path4/'sample_tr.csv', sep=',', low_memory=False)
# data.rename(columns={'sample': 'individualID'}, inplace=True)
# print(data)
# print(sample_tr)
# sample = sample_tr['sample'].tolist()
# print(sample)
# print(len(sample))
# positive = []
# negative = []
# # mid = []
# for i in range(351):
#     if data.individualID.iloc[i] not in sample and data.disease_type.iloc[i] == 1.0:
#         positive.append([data.individualID.iloc[i], data.cogdx.iloc[i], data.disease_type.iloc[i]])
#     if data.individualID.iloc[i] not in sample and data.disease_type.iloc[i] == 0.0:
#         negative.append([data.individualID.iloc[i], data.cogdx.iloc[i], data.disease_type.iloc[i]])
#     # if data.individualID.iloc[i] not in sample and data.disease_type_3.iloc[i] == 2.0:
#     #     mid.append([data.individualID.iloc[i], data.cogdx.iloc[i], data.disease_type_3.iloc[i]])
# positive = pd.DataFrame(positive)
# negative = pd.DataFrame(negative)
# # mid = pd.DataFrame(mid)
# positive.rename(columns={0: 'sample', 1: 'cogdx', 2: 'disease_type'}, inplace=True)
# negative.rename(columns={0: 'sample', 1: 'cogdx', 2: 'disease_type'}, inplace=True)
# # mid.rename(columns={0: 'sample', 1: 'cogdx', 2: 'disease_type_3'}, inplace=True)
# sample1 = pd.concat((positive, negative)).reset_index()
# sample1.drop(columns=['index'], axis=1, inplace=True)
# sample1 = shuffle(sample1)
# print(sample1)
# sample1.to_csv(path4/'sample_te.csv', index=False, sep=',')


# 分出三个组学数据的训练集测试集并对齐样本ID
# train = pd.read_csv(path4/'sample_tr.csv', sep=',', low_memory=False)
# test = pd.read_csv(path4/'sample_te.csv', sep=',', low_memory=False)
# dna = pd.read_csv(path4/'final_DNAmethylation_2.csv', sep=',', low_memory=False)
# mrna = pd.read_csv(path4/'final_mRNA_2.csv', sep=',', low_memory=False)
# mirna = pd.read_csv(path4/'final_miRNA_2.csv', sep=',', low_memory=False)
# dna.rename(columns={'TargetID': 'sample'}, inplace=True)
# mrna.rename(columns={'Genes': 'sample'}, inplace=True)
# mirna.rename(columns={'gene_id': 'sample'}, inplace=True)
#
# print(train)
# print(test)
# print(dna)
# print(mrna)
# print(mirna)
#
# dna_tr = pd.merge(train, dna, on='sample')
# dna_tr.drop(columns=['cogdx', 'disease_type_x'], inplace=True)
# dna_tr.rename(columns={'disease_type_y': 'disease_type'}, inplace=True)
# mirna_tr = pd.merge(train, mirna, on='sample')
# mirna_tr.drop(columns=['cogdx', 'disease_type_x'], inplace=True)
# mirna_tr.rename(columns={'disease_type_y': 'disease_type'}, inplace=True)
# mrna_tr = pd.merge(train, mrna, on='sample')
# mrna_tr.drop(columns=['cogdx', 'disease_type_x'], inplace=True)
# mrna_tr.rename(columns={'disease_type_y': 'disease_type'}, inplace=True)
# print(dna_tr)
# print(mirna_tr)
# print(mrna_tr)
#
# dna_tr.to_csv(path4/'DNA_tr.csv', index=False, sep=',')
# mirna_tr.to_csv(path4/'miRNA_tr.csv', index=False, sep=',')
# mrna_tr.to_csv(path4/'mRNA_tr.csv', index=False, sep=',')


# 训练集feature selection
# ANOVA('mRNA_tr.csv', 'sample')
# mean_var('final_miRNA_2.csv', 'gene_id', 0, 0.109)


# 分出测试集
# features = pd.read_csv(path6 / '3_featname.csv', sep=',', low_memory=False)
# test = pd.read_csv(path4 / 'miRNA_te.csv', sep=',', low_memory=False)
# print(test)
# print(features)
# a = 'hsa-let-7a'
# features = features[a].tolist()
# features.insert(0, a)
# print(features)
# test = test[features]
# test = normalize(test)
# print(test)
# test.to_csv(path5 / '3_te.csv', sep=',', index=False, header=False)


# 标签
# data1 = pd.read_csv(path4 / 'sample_tr.csv', sep=',', index_col=None, low_memory=False)
# a = data1.disease_type
# a.to_csv(path5 / 'labels_tr.csv', sep=',', index=False, header=False)

# 主体end


# with open(path4/'sample.txt', 'r') as f:
#     sample = f.readline()
# sample = sample[1:-1].split('\', \'')
# sample = {'individualID': sample}
# sample = pd.DataFrame(sample)
#
# print(sample)
#
# disease_type = pd.merge(data3, sample, on='individualID')
# disease_type[['cogdx', 'individualID']] = disease_type[['individualID', 'cogdx']]
# disease_type.rename(columns={'cogdx': 'individualID', 'individualID': 'disease_type'}, inplace=True)
# print(disease_type)
# disease_type.to_csv(path4/'disease_type.csv', sep='\t', index=False)
# data2 = pd.merge(data, data1, on='TargetID')

# data1.drop(columns=['Methyl27_Loci'], axis=1, inplace=True)
# data1.to_csv(path2/'27k_DNAmethylation.csv', sep='\t')
# data = data.drop(columns=['TargetID']).values
# data = data.T
# data = pd.DataFrame(data)
# data.drop('tracking_id', axis=1, inplace=True)
# data['R5693901'] = pd.Series([], dtype='float64')
# for i, v in data.iterrows():
#     data['R5693901'].iloc[i] = round((data['R5693901(1)'].iloc[i] + data['R5693901(2)'].iloc[i] + data['R5693901(3)'].iloc[i]) / 3, 2).copy()
# data.drop(['R5693901(1)', 'R5693901(2)', 'R5693901(3)'], axis=1, inplace=True)
# data1 = data1[data1.TargetID.isin(sample)]
# print(data1)
# data1.to_csv(path2/'final_DNAmethylation_2.csv', sep='\t', index=0)
# feat_name.to_csv(path2/'2_featname.csv', sep='\t', index=0, header=0)

# data1 = pd.read_csv(path4/'sample_shuffle.csv', sep='\t', low_memory=False)
# test = data1[:106]
# train = data1[106:]
# samples_te = pd.DataFrame(test['individualID'])
# labels_te = pd.DataFrame(test['is_AD'])
# samples_tr = pd.DataFrame(train['individualID'])
# labels_tr = pd.DataFrame(train['is_AD'])
# print(labels_te)
# print(labels_tr)
# print(samples_te)
# print(samples_tr)
# labels_te.to_csv(path4/'labels_te.csv', sep='\t', header=False, index=0)
# labels_tr.to_csv(path4/'labels_tr.csv', sep='\t', header=False, index=0)
# samples_tr.to_csv(path4/'sample_tr.csv', sep='\t', index=0)
# samples_te.to_csv(path4/'sample_te.csv', sep='\t', index=0)


# data1 = pd.read_csv(path4/'not_AD_sample.csv', sep='\t', low_memory=False)
# print(data1)
# data = data.T
# sample = data.columns.tolist()
# data1 = data1[data1.individualID.isin(sample)]
# print(data1)
# data3.to_csv(path4/'sample_shuffle.csv', sep='\t', index=0)


# mRNA = pd.read_csv(path6/'2_featname.csv', sep='\t', low_memory=False, names=[0])
# mrna = pd.read_csv(path4/'27k_DNAmethylation.csv', sep='\t', low_memory=False)
# te = pd.read_csv(path4/'sample_te.csv', sep='\t', low_memory=False)
# tr = pd.read_csv(path4/'sample_tr.csv', sep='\t', low_memory=False)
# mRNA = mRNA[0].tolist()
# mRNA.insert(0, 'TargetID')
# mrna = mrna[mRNA]
# mrna_te = mrna[mrna.TargetID.isin(te['individualID'].tolist())]
# mrna_te.drop(['TargetID'], axis=1, inplace=True)
# mrna_tr = mrna[mrna.TargetID.isin(tr['individualID'].tolist())]
# mrna_tr.drop(['TargetID'], axis=1, inplace=True)
# mrna_te = pd.DataFrame(normalize(np.array(mrna_te)))
# mrna_tr = pd.DataFrame(normalize(np.array(mrna_tr)))
# print(mrna_te)
# print(mrna_tr)
# mrna_tr.to_csv(path5/'1_tr.csv', sep=',', index=False, header=False)
# mrna_te.to_csv(path5/'1_te.csv', sep=',', index=False, header=False)
