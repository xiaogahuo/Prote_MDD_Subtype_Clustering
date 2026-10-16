from SUtils import *


def social_statistics(exp, target_exp, category_cols, continue_cols):
    df_NC, df_MDD = pd.read_csv(data_path + f"{exp}_NC.csv", low_memory=False), pd.read_csv(data_path + f"{exp}_MDD.csv", low_memory=False)
    df_NC['final_cluster'] = 0
    df_Y = pd.read_csv(result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(target_exp,"NMF", 2, 2), low_memory=False, usecols=['eid', 'final_cluster'])
    df_MDD = pd.merge(df_MDD, df_Y, how='inner', on='eid', sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)

    df = pd.concat([df_MDD, df_NC])
    df_1_VS_2 = df[df['final_cluster'].isin([1,2])]
    table = build_table_statistics(df_1_VS_2, 'final_cluster', [1, 2], category_cols, continue_cols)
    out_file = os.path.join(result_path, f'Table_demographics_{exp}_1_VS_2_clustering.xlsx')
    table.to_excel(out_file, index=False)

    df_0_VS_1 = df[df['final_cluster'].isin([0,1])]
    table = build_table_statistics(df_0_VS_1, 'final_cluster', [0, 1], category_cols, continue_cols)
    out_file = os.path.join(result_path, f'Table_demographics_{exp}_0_VS_1_clustering.xlsx')
    table.to_excel(out_file, index=False)

    df_0_VS_2 = df[df['final_cluster'].isin([0,2])]
    table = build_table_statistics(df_0_VS_2, 'final_cluster', [0, 2], category_cols, continue_cols)
    out_file = os.path.join(result_path, f'Table_demographics_{exp}_0_VS_2_clustering.xlsx')
    table.to_excel(out_file, index=False)


if __name__ == '__main__':
    category_cols = cat_cols + cate100057_cols[1:]
    continue_cols = [cate100057_cols[0]] + conti_cols
    # social_statistics('Diag_Discovery', 'Diag_Discovery', category_cols, continue_cols)
    # social_statistics('Diag_Test', 'Diag_Test', category_cols, continue_cols )

    social_statistics('Diag_self_Discovery', 'Diag_Discovery', category_cols, continue_cols)
    social_statistics('Diag_self_Test', 'Diag_Test', category_cols, continue_cols)
    social_statistics('Diag_drug_Discovery', 'Diag_Discovery', category_cols, continue_cols)
    social_statistics('Diag_drug_Test', 'Diag_Test', category_cols, continue_cols)

