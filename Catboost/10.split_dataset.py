from functools import reduce
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from SUtils import *
import os


def load_binary_df(exp, target_exp, clu_list):
    """读取 exp 的 MDD+HC, 按 [0](HC) vs clu_list(病例) 过滤并二值化 final_cluster"""
    rst_file = result_path + "{}/NMF/2_components_cluster_rst_2.csv".format(target_exp)
    df_Y = pd.read_csv(rst_file, low_memory=False, usecols=['eid', 'final_cluster'])
    df_MDD = pd.read_csv(data_path + "{}_MDD.csv".format(exp), low_memory=False)
    df_MDD = pd.merge(df_MDD, df_Y, on='eid', how='inner')

    df_NC = pd.read_csv(data_path + "{}_NC.csv".format(exp), low_memory=False)
    df_NC['final_cluster'] = 0
    df = pd.concat([df_MDD, df_NC], ignore_index=True)

    # 0 vs 1: 0=对照(HC), 1=病例(亚型)
    df = df[df['final_cluster'].isin([0] + clu_list)].copy()
    df['final_cluster'] = df['final_cluster'].isin(clu_list).astype(int)  # 将 clu_list统一修改为1
    return df


def split_dataset(train_exp, target_train_exp, test_exp, target_test_exp):
    """Discovery 切 5 折(用于网格搜索) + 全集(最终模型训练集); Test 作为独立外部测试集"""
    for clu_list, dataset in [([1, 2], '0_VS_12'), ([1], '0_VS_1'), ([2], '0_VS_2')]:
        df_train = load_binary_df(train_exp, target_train_exp, clu_list)   # Discovery 全集(二值)
        df_test = load_binary_df(test_exp, target_test_exp, clu_list)     # Test 全集(二值)

        base = data_path + "cross_valid/{}_{}/".format(train_exp, dataset)
        if not os.path.exists(base):
            os.makedirs(base)

        # 最终模型训练集 = 全部 Discovery 样本; 外部测试集 = 全部 Test 样本
        df_train.to_csv(base + "full_train.csv", index=None)
        df_test.to_csv(base + "test.csv", index=None)

        # 5 折交叉验证划分(固定随机种子, 分层)
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        X, y = df_train, df_train['final_cluster']
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            fold_dir = base + "fold{}/".format(fold)
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            X.iloc[tr_idx].to_csv(fold_dir + "train.csv", index=None)
            X.iloc[va_idx].to_csv(fold_dir + "valid.csv", index=None)
        print(train_exp, dataset, 'train:', df_train.groupby('final_cluster').size().to_dict(),
              'test:', df_test.groupby('final_cluster').size().to_dict())


if __name__ == '__main__':
    # split_dataset('Diag_Discovery', 'Diag_Discovery', 'Diag_Test', 'Diag_Test')
    split_dataset('Diag_self_Discovery', 'Diag_Discovery', 'Diag_self_Test', 'Diag_Test')
    split_dataset('Diag_drug_Discovery', 'Diag_Discovery', 'Diag_drug_Test', 'Diag_Test')
