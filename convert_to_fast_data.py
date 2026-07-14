import os
import torch
import pickle
from tqdm import tqdm
import numpy as np
import sys

# 导入原有的工具函数
from utils.net_utils import get_mask


def convert_to_pt(src_pkl, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"正在读取: {src_pkl}")
    with open(src_pkl, 'rb') as f:
        data_all = pickle.load(f)

    num_samples = len(data_all[0])
    print(f"开始转换 {num_samples} 个样本...")

    for i in tqdm(range(num_samples)):
        # 使用 torch.as_tensor().float() 或 .long() 替代原来的显式 FloatTensor
        sample = {
            'RNA_repre': torch.as_tensor(data_all[0][i]).float(),
            'Mol_graph': data_all[1][i],
            'RNA_Graph': data_all[2][i],
            'RNA_feats': torch.as_tensor(data_all[3][i]).long(),
            'RNA_C4_coors': torch.as_tensor(data_all[4][i]).float(),
            'RNA_coors': torch.as_tensor(data_all[5][i]).float(),
            'Mol_feats': torch.as_tensor(data_all[6][i]).long(),
            'Mol_coors': torch.as_tensor(data_all[7][i]).float(),
            'Mol_LAS': torch.as_tensor(data_all[8][i]).float(),  # 这里修复了报错
            'label': torch.as_tensor([data_all[9][i]]).float(),

            # 预存 Mask
            'RNA_repre_mask': torch.as_tensor(get_mask([data_all[0][i]])[0]).float(),
            'RNA_feats_mask': torch.as_tensor(get_mask([data_all[3][i]])[0]).bool(),
            'Mol_coors_mask': torch.as_tensor(get_mask([data_all[7][i]])[0]).bool(),
        }
        torch.save(sample, os.path.join(save_dir, f"sample_{i}.pt"))

if __name__ == "__main__":
    # 执行转换
    # 你可以根据需要修改路径
    # convert_to_pt("./data/Robin/ran/train_data.pkl", "./data/Robin/both/train_pt")
    # convert_to_pt("./data/Robin/both/valid_data.pkl", "./data/Robin/both/valid_pt")
    # convert_to_pt("./data/Robin/both/test_data.pkl", "./data/Robin/both/test_pt")
    convert_to_pt("./data/Biosensor/random/train_data.pkl", "./data/Biosensor/random/train_pt")
    convert_to_pt("./data/Biosensor/random/valid_data.pkl", "./data/Biosensor/random/valid_pt")
    convert_to_pt("./data/Biosensor/random/test_data.pkl", "./data/Biosensor/random/test_pt")