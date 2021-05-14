import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import csv
import pandas as pd


path='ctbpred.csv'
data=pd.read_csv(path)
# 短句预测全对的更多
#max length限制
#分词错误和标注错误分开统计
#预测不存在tag的

for i in data:
    score, ref, pred = i[0], i[1], i[2]