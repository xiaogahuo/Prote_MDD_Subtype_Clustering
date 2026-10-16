import pandas as pd

from SUtils import *
from functools import reduce
from scipy.stats.mstats import kruskal

def trans_genotyping_array(x):
    if -11 <= x <= -1:
        return 1   # 'BiLEVE'
    elif 1 <= x <= 95:
        return 2  # 'Axiom'
    else:
        return None  # 或者根据需要返回其他值

covar_num = 6  # <=41

def merge_data(exp):
    cols = load_prote_sig_cols()
    df_MDD, df_NC = pd.read_csv(data_path + f"{exp}_MDD.csv", low_memory=False), pd.read_csv(data_path + f"{exp}_NC.csv", low_memory=False)
    df_NC['final_cluster'] = 0
    df_Y = pd.read_csv(result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(exp, "NMF", 2, 2), low_memory=False,
                       usecols=['eid', 'final_cluster'])
    df_MDD = pd.merge(df_MDD, df_Y, how='inner', on='eid', sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)
    df = pd.concat([df_MDD, df_NC], ignore_index=True)
    df[['eid', 'final_cluster']+cols+covr_cols+cate100057_cols].to_csv(data_path + f"Linear_Regression_Prote_Dataset_{exp}.csv", index=False)

    df_PRS_ALL = [df[['eid', 'final_cluster']+cols+covr_cols+cate100057_cols]]
    for f in ['5e08', '1e07', '5e07', '1e06', '5e06', '1e05', '5e05', '1e04', '5e04', '1e03', '5e03', '1e02', '5e02']:
        df_PRS = pd.read_csv(PRS_path + "{}_PRS.sscore".format(f), sep=r'[ \t]', engine='python', usecols=['#FID', 'SCORE1_AVG'])
        mean, std = df_PRS['SCORE1_AVG'].mean(), df_PRS['SCORE1_AVG'].std()
        df_PRS['SCORE1_AVG'] = (df_PRS['SCORE1_AVG'] - mean) / std
        df_PRS.rename(columns={'#FID': 'eid', 'SCORE1_AVG': 'SCORE_'+f}, inplace=True)
        df_PRS_ALL.append(df_PRS)
    df_PRS_ALL = reduce(lambda left, right: pd.merge(left, right, on=['eid'], how='inner'), df_PRS_ALL)
    df_MDS = pd.read_csv(root_path + "Sample_QC.csv")
    PCA_original_cols, PCA_new_cols = ["22009-0." + str(i) for i in range(1, covar_num)], ["PCA" + str(i) for i in range(1, covar_num)]
    df_MDS.rename(columns=dict(zip(PCA_original_cols, PCA_new_cols)), inplace=True)
    df_MDS['genotyping_array'] = df_MDS['22000-0.0'].apply(trans_genotyping_array)
    df_MDS = df_MDS[['eid', 'genotyping_array']+PCA_new_cols]
    df_PRS_ALL = reduce(lambda left, right: pd.merge(left, right, on=['eid'], how='inner'), [df_PRS_ALL, df_MDS])
    df_PRS_ALL.to_csv(data_path + f"Linear_Regression_PRS_Dataset_{exp}.csv", index=False)

    # cols_MRI = pd.read_excel(path + "Cate192_cols.xlsx")['FieldID'].tolist()
    # print(cols_MRI)
    cols_MRI_192 = [str(c) + '-2.0' for c in cols_MRI_c192_Area+cols_MRI_c192_Thickness+cols_MRI_c192_Volume]
    df_MRI_192 = pd.read_csv(path + "Cate_192.csv", low_memory=False, usecols=['eid']+cols_MRI_192)
    df_MRI_192.dropna(thresh=2, inplace=True)
    print("df_MRI_192: {}".format(df_MRI_192.shape[0]))

    cols_MRI_135 = [str(c) + '-2.0' for c in cols_MRI_c135_MD+cols_MRI_c135_FA]
    df_MRI_135 = pd.read_csv(path + "Cate_135.csv", low_memory=False, usecols=['eid']+cols_MRI_135)
    df_MRI_135.dropna(thresh=2, inplace=True)
    print("df_MRI_135: {}".format(df_MRI_135.shape[0]))

    cols_MRI_110 = [str(c) + '-2.0' for c in cols_MRI_c1102_subcortical_volume]
    df_MRI_110 = pd.read_csv(path + "Cate_110.csv", low_memory=False, usecols=['eid'] + cols_MRI_110)
    df_MRI_110.dropna(thresh=2, inplace=True)
    print("df_MRI_110: {}".format(df_MRI_110.shape[0]))

    # df_rsfMRI = pd.read_csv(path + "25753_2_0.csv", low_memory=False, usecols=['eid']+cols_MRI_field_25753)
    # print("df_rsfMRI: {}".format(df_rsfMRI.shape[0]))
    # grouped = df_rsfMRI.groupby('final_cluster').size()
    # print(grouped)

    df_MRI = reduce(lambda left, right: pd.merge(left, right, on=['eid'], how='inner'), [df[['eid', 'final_cluster']+covr_cols+cate100057_cols], df_MRI_192, df_MRI_135, df_MRI_110])
    print("df_MRI: {}".format(df_MRI.shape[0]))
    for col in cols_MRI_192+cols_MRI_135+cols_MRI_110:
        df_MRI[col] = (df_MRI[col] - df_MRI[col].mean())/df_MRI[col].std()
    df_MRI.fillna(0, inplace=True)  # 空值补零- 即均值
    cols_MRI = cols_MRI_192+cols_MRI_135+cols_MRI_110
    new_cols = ['f' + c[:-4] for c in cols_MRI]
    print(new_cols)
    df_MRI.rename(columns=dict(zip(cols_MRI, new_cols)), inplace=True)
    grouped = df_MRI.groupby('final_cluster').size()
    print(grouped)
    df_MRI.to_csv(data_path + f"Linear_Regression_MRI_Dataset_{exp}.csv", index=False)


def plot_bar_PRS(exp):
    df = pd.read_csv(data_path + f"Linear_Regression_PRS_Dataset_{exp}.csv", low_memory=False)
    df['final_cluster'] = df['final_cluster'].replace({0: 'HC', 1: 'Subtpye1', 2: 'Subtpye2'})
    PRS_cols = ['SCORE_5e08', 'SCORE_1e07', 'SCORE_5e07', 'SCORE_1e06', 'SCORE_5e06', 'SCORE_1e05', 'SCORE_5e05', 'SCORE_1e04', 'SCORE_5e04', 'SCORE_1e03', 'SCORE_5e03', 'SCORE_1e02', 'SCORE_5e02']
    new_cols = ['5e-08', '1e-07', '5e-07', '1e-06', '5e-06', '1e-05', '5e-05', '1e-04', '5e-04', '1e-03', '5e-03', '0.01', '0.05']
    df.rename(columns=dict(zip(PRS_cols, new_cols)), inplace=True)
    print(df.head())
    for col in new_cols:
        df[col] = (df[col] - df[col].mean())/df[col].std()
    plot_bar(df, f'PRS_{exp}', new_cols, 13, 4, ['HC', 'Subtpye1', 'Subtpye2'], 'PRS_Zscore')


def social_stats(exp, group_col, group_order, category_cols, continue_cols, out_path=result_path):
    import os
    os.makedirs(out_path, exist_ok=True)

    df = pd.read_csv(data_path + exp + '.csv')
    df = df[df['final_cluster'].isin(group_order)]
    print('---------------exp: {}, number: {}------------------------'.format(exp, df.shape[0]))
    print(df.columns.tolist())
    table = build_table_statistics(df, group_col, group_order, category_cols, continue_cols)
    out_file = os.path.join(out_path, f'Table1_demographics_{exp}_{group_col}.xlsx')
    table.to_excel(out_file, index=False)
    print(table.to_string(index=False))
    print('已保存: {}'.format(out_file))
    print()


if __name__ == '__main__':
    merge_data('Diag_Discovery')
    merge_data('Diag_Test')
    plot_bar_PRS('Diag_Discovery')
    plot_bar_PRS('Diag_Test')

    # 社会人口学数据统计
    category_cols = cat_cols + cate100057_cols[1:]
    continue_cols = [cate100057_cols[0]] + conti_cols
    social_stats('Linear_Regression_PRS_Dataset_Diag_Discovery', 'final_cluster', [0,1,2], category_cols, continue_cols)
    social_stats('Linear_Regression_PRS_Dataset_Diag_Test' , 'final_cluster', [0,1,2], category_cols, continue_cols)

    social_stats('Linear_Regression_MRI_Dataset_Diag_Discovery', 'final_cluster', [0, 1, 2], category_cols, continue_cols)
    social_stats('Linear_Regression_MRI_Dataset_Diag_Test', 'final_cluster', [0, 1, 2], category_cols, continue_cols)



