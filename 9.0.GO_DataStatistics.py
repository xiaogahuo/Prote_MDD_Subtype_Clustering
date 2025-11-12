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


def format_GO_statistics(exp):
    bonferroni_file = result_path + f"t_test_result_bonferroni_{exp}.csv"
    cols_prote = pd.read_csv(bonferroni_file, low_memory=False)['col'].tolist()
    df_prote = pd.read_csv(path + "proteomics.csv", usecols=['eid']+cols_prote)

    df_NC = pd.read_csv(data_path + f"{exp}_NC.csv", low_memory=False, usecols=['eid'])
    df_NC['final_cluster'] = 0
    df_NC = reduce(lambda left, right: pd.merge(left, right, on=['eid'], how='inner'), [df_NC, df_prote])
    df_MDD = pd.read_csv(result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(exp, "NMF", 2, 2), low_memory=False, usecols=['eid', 'final_cluster'])
    df_MDD = reduce(lambda left, right: pd.merge(left, right, on=['eid'], how='inner'), [df_MDD, df_prote])
    df_MDD_clst1, df_MDD_clst2 = df_MDD[df_MDD['final_cluster'] == 1], df_MDD[df_MDD['final_cluster'] == 2]
    df_clst1, df_clst2 = pd.concat([df_NC, df_MDD_clst1]), pd.concat([df_NC, df_MDD_clst2])

    f_clst1, f_clst2 = open(result_path + f"GO_sig_cols_clst1_{exp}.csv", "w"), open(result_path + f"GO_sig_cols_clst2_{exp}.csv", "w")
    print("col,logFC,NC_mean,Clst1_mean,P.Value", file=f_clst1), print("col,logFC,NC_mean,Clst2_mean,P.Value", file=f_clst2)
    for prote, i in zip(cols_prote, range(len(cols_prote))):
        print('i:{}, prote:{}'.format(i, prote))
        mean_NC, mean_clst1, mean_clst2 = df_NC[prote].mean(), df_MDD_clst1[prote].mean(), df_MDD_clst2[prote].mean()
        logFC_clst1, logFC_clst2 = np.log2((mean_clst1)/(mean_NC)), np.log2((mean_clst2)/(mean_NC))
        pvalue_clst1, pvalue_clst2 = kwtest(prote, 'final_cluster', df_clst1), kwtest(prote, 'final_cluster', df_clst2)
        print("{},{},{},{},{}".format(prote, logFC_clst1, mean_NC, mean_clst1,pvalue_clst1), file=f_clst1)
        print("{},{},{},{},{}".format(prote, logFC_clst2, mean_NC, mean_clst2,pvalue_clst2), file=f_clst2)


if __name__ == '__main__':
    format_GO_statistics('Diag')




