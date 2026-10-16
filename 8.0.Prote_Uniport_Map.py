import pandas as pd

from SUtils import *
from statsmodels.stats.multitest import fdrcorrection


def kwtest(feature_name, target_feature, df):
    print('kwtest结果: <' + feature_name + "> on <" + target_feature + ">")
    # print(df.drop_duplicates(subset=[target_feature], keep='first')[target_feature])
    target_feature_value_list = df.drop_duplicates(subset=[target_feature], keep='first')[target_feature].tolist()

    kw_data = []
    for value in target_feature_value_list:
        feature_name_value_list = df[df[target_feature] == value][feature_name].tolist()
        kw_data.append(feature_name_value_list)
    statistic, pvalue = kruskalwallis(*kw_data) #  将kw_data中的值作为参数传入kruskalwallis中, 第1个值是第1个参数， 第2个值是第2个参数
    print("column: {}, statistic: {}, pvalue: {}".format(feature_name, statistic, pvalue))
    return pvalue


# def mapping_prote_uniprot():
#     df = pd.read_excel(path +'NEW蛋白组学词典.xlsx', engine='openpyxl', sheet_name='NEW蛋白组学词典', usecols=['Assay', 'UniProt'])
#     df_col = pd.DataFrame({'col': Prote_cols_all})
#     df_sig = pd.merge(df_col, df, left_on='col', right_on='Assay', how='left')
#     df_sig.to_csv(result_path+"ALL_Prote_Uniprot.csv", index=False)
#     print(df_sig.shape)


def format_GSEA_data(exp):
    cols = Prote_cols_all
    df_NC = pd.read_csv(data_path + f"{exp}_NC.csv", low_memory=False, usecols=['eid']+cols)
    df_MDD = pd.read_csv(data_path + f"{exp}_MDD.csv", low_memory=False, usecols=['eid'] + cols)
    df_NC['final_cluster'] = 0
    df_Y = pd.read_csv(result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(exp, "NMF", 2, 2),
                         low_memory=False, usecols=['eid', 'final_cluster'])
    df_MDD = reduce(lambda left, right: pd.merge(left, right, on=['eid'], how='inner'), [df_MDD, df_Y])
    df_MDD_clst1, df_MDD_clst2 = df_MDD[df_MDD['final_cluster'] == 1], df_MDD[df_MDD['final_cluster'] == 2]
    df_clst1, df_clst2 = pd.concat([df_NC, df_MDD_clst1]), pd.concat([df_NC, df_MDD_clst2])

    # df = pd.read_excel(path +'NEW蛋白组学词典.xlsx', engine='openpyxl', sheet_name='NEW蛋白组学词典', usecols=['Assay', 'UniProt'])
    df = pd.read_csv(path +'ALL_Prote_Uniprot.csv', usecols=['Assay', 'UniProt'])
    dicts = dict(zip(df['Assay'].tolist(), df['UniProt'].tolist()))

    f_clst1, f_clst2 = open(result_path + f"GO_KEGG_sig_cols_clst1_{exp}.csv", "w"), open(
        result_path + f"GO_KEGG_sig_cols_clst2_{exp}.csv", "w")
    print("col,foldchange,NC_mean,Clst1_mean,P.Value,GeneName", file=f_clst1), print(
        "col,foldchange,NC_mean,Clst2_mean,P.Value,GeneName", file=f_clst2)
    for prote, i in zip(cols, range(len(cols))):
        print('i:{}, prote:{}'.format(i, prote))
        mean_NC, mean_clst1, mean_clst2 = df_NC[prote].mean(), df_MDD_clst1[prote].mean(), df_MDD_clst2[prote].mean()
        Clst1_minus_NC, Clst2_minus_NC = mean_clst1-mean_NC, mean_clst2-mean_NC
        pvalue_clst1, pvalue_clst2 = kwtest(prote, 'final_cluster', df_clst1), kwtest(prote, 'final_cluster', df_clst2)
        print("{},{},{},{},{},{}".format(prote, Clst1_minus_NC, mean_NC, mean_clst1, pvalue_clst1, dicts.get(prote, "")),
              file=f_clst1)
        print("{},{},{},{},{},{}".format(prote, Clst2_minus_NC, mean_NC, mean_clst2, pvalue_clst2, dicts.get(prote, "")),
              file=f_clst2)


def sig(exp, clust):
    csv_file = result_path + f"GO_KEGG_sig_cols_{clust}_{exp}.csv"
    df = pd.read_csv(csv_file)
    bool_r, p_fdr = fdrcorrection(df['P.Value'].tolist(), alpha=0.05, method='indep', is_sorted=False)
    df['adj.P.Val'] = pd.Series(p_fdr)
    df.to_csv(csv_file, index=None)


if __name__ == '__main__':
    format_GSEA_data('Diag_Discovery')
    format_GSEA_data('Diag_Test')

    sig('Diag_Discovery', 'clst1')
    sig('Diag_Discovery', 'clst2')
    sig('Diag_Test', 'clst1')
    sig('Diag_Test', 'clst2')




