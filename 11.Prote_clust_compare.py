from scipy.stats.mstats import kruskal
import pandas as pd
from SUtils import *
import math


def compare_KW(df, fname, exp):
    f = open(result_path + f"{exp}_compare_kw_result_{fname}.csv", "w")
    header_str = "col,statistic,pvalue,M_all,STD_all"
    div_feature_value_list = sorted(df.drop_duplicates(subset=['final_cluster'], keep='first')['final_cluster'].tolist())
    for c in div_feature_value_list:
        header_str = header_str + ',M{},STD{}'.format(c,c)
    print(header_str, file=f)
    print(header_str)
    cols =Prote_cols_all+ [c+'-0.0' for c in ['1160','4609','4620','5375','5386','20127','2050','2060','2070','2080']] + cate17518_cols + conti_cols + ['PHQ_Total', 'PHQ']
    # cols =[c+'-0.0' for c in ['1160','4609','4620','5375','5386','20127']]
    for i, col in zip(range(len(cols)), cols):
        print("clust_N:{}, index:{}, col:{}".format(fname, i, col))
        # 按final_cluster列的值对col列进行分组
        grp_list =[]
        for c in div_feature_value_list:
            grp_list.append(df[df['final_cluster']==c][col].to_list())
        statistic, pvalue = kruskal(*grp_list)
        M_all, STD_all = df[col].mean(), df[col].std(ddof=1)
        val_str = col + ',' + str(statistic) + ',' + str(pvalue) + ',' + str(M_all) + ',' + str(STD_all)
        for c in div_feature_value_list:
            M, STD = df[df['final_cluster']==c][col].mean(), df[df['final_cluster']==c][col].std(ddof=1)
            val_str = val_str + ',' + str(M) + ',' + str(STD)
        print(val_str, file=f)
        
        
def compare_Chi2(df, fname, exp):
    f = open(result_path + f"{exp}_compare_kafang_result_{fname}.csv", "w")
    header_str = "col,statistic,pvalue,degree_freedom,cnt_feature_values,chi2_data"
    print(header_str, file=f)
    cate_100060=[c+'-0.0' for c in ['20123','20124','20125','1920','1930','1940','1950','1960','1970','1980','1990','2000','2010','2020','2030','2040','2090','2100','4598','4631','4642','4653','4526','4548','4559','4570','4581','4537','6156','5663','5674','6145','10721','20122','20126']]
    cols =cat_cols+ cate100042_cols + cate100057_cols[1:] + cate_100060 + cate100061_cols + cate100065_cols + cate145_cols
    for i, col in zip(range(len(cols)), cols):
        print("--------clust_N:{}, index:{}, col:{}--------".format(fname, i, col))
        null_ratio = df[col].isnull().mean()
        print("null_ratio: {}".format(null_ratio))
        if null_ratio >= 0.5:
            continue
        print('卡方检验结果: <' + col + "> on < final_cluster >")
        div_feature_value_list = sorted(df.drop_duplicates(subset=['final_cluster'], keep='first')['final_cluster'].tolist())
        cnt_feature_value_list = df.drop_duplicates(subset=[col], keep='first')[col].tolist()
        cnt_feature_value_series = pd.Series(cnt_feature_value_list)
        cnt_feature_value_series = cnt_feature_value_series.dropna()
        cnt_feature_value_list = sorted(cnt_feature_value_series.to_list())
        print("{}: {}".format('final_cluster', div_feature_value_list))
        print("{}: {}".format(col, cnt_feature_value_list))

        kf_data = []
        for div in div_feature_value_list:
            c_g =[]
            for value in cnt_feature_value_list:
                count = df[(df['final_cluster'] == div) & (df[col] == value)].shape[0]
                c_g.append(count)
            kf_data.append(c_g)
        print(kf_data)
        statistic, pvalue, dof, expected_freq = chi2_contingency(kf_data)
        print("{},{},{},{},{},{}".format(col, statistic, pvalue, dof, cnt_feature_value_list, kf_data), file=f)


def bar_biochem_plot(df, exp):
    df['final_cluster'] = df['final_cluster'].replace({0: 'HC', 1: 'Subtpye1', 2: 'Subtpye2'})
    df_biochem = df[cate17518_cols+['final_cluster']]
    df_biochem.rename(columns=dict(zip(cate17518_cols, cate17518_new_cols)), inplace=True)

    for col in cate17518_new_cols:
        df_biochem[col] = (df_biochem[col] - df_biochem[col].mean())/df_biochem[col].std()
    plot_bar(df_biochem, f'biochem_1_{exp}', ['IGF-1','ALP','ALT','APOA1','AST','CRP','CYSC','GGT','HbA1c','HDL','SHBG','TRIG','UA','VITD'], 12, 4, ['HC', 'Subtpye1', 'Subtpye2'], 'Z-score')
    plot_bar(df_biochem, f'biochem_2_{exp}', ['ALB','APOB','DBIL','UREA','CALC','CHOL','CREA','GLU','LDL','LP-a','PHOS','TBIL','TEST','TP'], 12, 4, ['HC', 'Subtpye1', 'Subtpye2'], 'Z-score')


def plot_violin(df, file_name, feature_lst, width, high, hue_order, y_label):
    df_violin = []
    for feature in feature_lst:
        df_sub = df[[feature, 'final_cluster']].copy()
        df_sub['Features'] = feature
        df_sub.rename(columns={feature: y_label}, inplace=True)
        # print(df_sub.head())
        df_violin.append(df_sub)
    df_violin = pd.concat(df_violin, ignore_index=True)
    # print("before: {}".format(df_violin.shape))
    print(df_violin.head())

    plt.figure(figsize=(width, high))
    ax = sns.violinplot(hue="final_cluster", hue_order=hue_order, y=y_label, x="Features", data=df_violin,
                palette=palette, width=.9, gap=0.1) # errorbar=None
    ymax = df_violin[y_label].max()
    ax.set_ylim(-0.4, ymax*1.2)
    ax.tick_params(axis='x', labelrotation=0, labelsize=8)
    ax.tick_params(axis='y', labelrotation=0, labelsize=8)
    plt.legend(loc='upper center', bbox_to_anchor=(1.47, 1.02))  # 使用这行代码把图例放到图外。
    plt.savefig(img_path + "Violin_{}.png".format(file_name), dpi=720)
    plt.clf()


def PHQ_plot(df, exp):
    df['final_cluster'] = df['final_cluster'].replace({0: 'HC', 1: 'Subtpye1', 2: 'Subtpye2'})
    df_PHQ = df[['20127-0.0', '2050-0.0', '2060-0.0', '2070-0.0', '2080-0.0', 'PHQ', 'PHQ_Total', 'final_cluster']]
    for col in ['20127-0.0', '2050-0.0', '2060-0.0', '2070-0.0', '2080-0.0', 'PHQ', 'PHQ_Total']:
        plot_violin(df_PHQ, col+f"_{exp}", [col], 3, 4, ['HC', 'Subtpye1', 'Subtpye2'], 'Score')


def prote_plot_cmp(exp):
    df_MDD, df_NC = pd.read_csv(data_path + f"{exp}_MDD.csv", low_memory=False), pd.read_csv(data_path + f"{exp}_NC.csv", low_memory=False)
    df_NC['final_cluster'] = 0
    df_Y = pd.read_csv(result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(exp,"NMF", 2, 2), low_memory=False, usecols=['eid', 'final_cluster'])
    df_MDD = pd.merge(df_MDD, df_Y, how='inner', on='eid', sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)
    # compare_KW(df_MDD, 'Clst1_VS_Clst2', exp)
    # compare_Chi2(df_MDD, 'Clst1_VS_Clst2', exp)
    df_NC_VS_Clst1 = pd.concat([df_NC, df_MDD[df_MDD['final_cluster'] == 1]])
    # compare_KW(df_NC_VS_Clst1, 'Clst1_VS_NC', exp)
    # compare_Chi2(df_NC_VS_Clst1, 'Clst1_VS_NC', exp)
    df_NC_VS_Clst1['final_cluster'] = df_NC_VS_Clst1['final_cluster'].replace({0: 'HC', 1: 'Subtpye1'})
    # plot_bar(df_NC_VS_Clst1, f'Clust1_ONLY_{exp}', Prote_cols_subtype1, 18, 4, ['HC', 'Subtpye1'], 'Z-score')
    # plot_bar(df_NC_VS_Clst1, f'Clust1_ALL_{exp}', Prote_cols_subtype1_and_2, 18, 4, ['HC', 'Subtpye1'], 'Z-score')

    df_NC_VS_Clst2 = pd.concat([df_NC, df_MDD[df_MDD['final_cluster'] == 2]])
    # compare_KW(df_NC_VS_Clst2, 'Clst2_VS_NC', exp)
    # compare_Chi2(df_NC_VS_Clst2, 'Clst2_VS_NC', exp)
    df_NC_VS_Clst2['final_cluster'] = df_NC_VS_Clst2['final_cluster'].replace({0: 'HC', 2: 'Subtpye2'})
    # plot_bar(df_NC_VS_Clst2, f'Clust2_ONLY_{exp}', Prote_cols_subtype2, 18, 4, ['HC', 'Subtpye2'], 'Z-score')
    # plot_bar(df_NC_VS_Clst2, f'Clust2_ALL_{exp}', Prote_cols_subtype1_and_2, 18, 4, ['HC', 'Subtpye2'], 'Z-score')

    df = pd.concat([df_NC, df_MDD])
    # compare_KW(df, 'ALL', exp)
    # compare_Chi2(df, 'ALL', exp)
    # bar_biochem_plot(df, exp)
    PHQ_plot(df, exp)

    # df_MDD['final_cluster'] = 1
    # df_ALL_Cmp = pd.concat([df_NC, df_MDD])
    # df_ALL_Cmp['final_cluster'] = df_ALL_Cmp['final_cluster'].replace({0: 'HC', 1: 'MDD'})
    # plot_bar(df_ALL_Cmp, f'ALL_Clust1_{exp}', Prote_cols_subtype1, 18, 4, ['HC', 'MDD'], 'Z-score')
    # plot_bar(df_ALL_Cmp, f'ALL_Clust2_{exp}', Prote_cols_subtype2, 18, 4, ['HC', 'MDD'], 'Z-score')
    # plot_bar(df_ALL_Cmp, f'ALL_ALL_{exp}', Prote_cols_subtype1_and_2, 18, 4, ['HC', 'MDD'], 'Z-score')


if __name__ == '__main__':
    prote_plot_cmp('Diag')
    prote_plot_cmp('DiagA')
    prote_plot_cmp('DiagB')


