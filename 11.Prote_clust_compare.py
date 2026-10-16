from scipy.stats.mstats import kruskal
import pandas as pd
from SUtils import *
from scipy import stats
from statsmodels.stats.multitest import multipletests


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

def load_top20_shap_prote(clust_cmp, top20_shap_prote):
    file_path = img_path+f"Diag_Discovery_{clust_cmp}/Diag_Discovery_{clust_cmp}_{top20_shap_prote}_SHAP.csv"
    print(file_path)
    df = pd.read_csv(file_path, low_memory=False)
    cols_top20 = df.iloc[:20]['key'].tolist()
    print(clust_cmp, top20_shap_prote, cols_top20)
    return cols_top20



def prote_plot(exp, top20_shap_prote):
    df_MDD, df_NC = pd.read_csv(data_path + f"{exp}_MDD.csv", low_memory=False), pd.read_csv(data_path + f"{exp}_NC.csv", low_memory=False)
    df_NC['final_cluster'] = 0
    df_Y = pd.read_csv(result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(exp,"NMF", 2, 2), low_memory=False, usecols=['eid', 'final_cluster'])
    df_MDD = pd.merge(df_MDD, df_Y, how='inner', on='eid', sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)
    df = pd.concat([df_MDD, df_NC])


    df_0_VS_1 = df[df['final_cluster'].isin([0,1])]
    df_0_VS_1['final_cluster'] = df_0_VS_1['final_cluster'].replace({0: 'HC', 1: 'Subtpye1'})
    # cols = load_top20_shap_prote('0_VS_1', top20_shap_prote)
    cols = ['RARRES2', 'GUSB', 'IL1RN', 'PIGR', 'ASGR1', 'NTRK3', 'NCAM1', 'NOTCH3', 'PRAP1', 'IGSF9', 'LRRN1', 'DPY30', 'CD83', 'ADAMTS15', 'HJV', 'CCER2', 'GFAP', 'IGSF8', 'ADM', 'CDHR5']
    print("0_VS_1:", cols)
    plot_bar(df_0_VS_1, f'Clust1_{exp}', cols, 18, 4, ['HC', 'Subtpye1'], 'Z-score')

    df_0_VS_2 = df[df['final_cluster'].isin([0,2])]
    df_0_VS_2['final_cluster'] = df_0_VS_2['final_cluster'].replace({0: 'HC', 2: 'Subtpye2'})
    # cols = load_top20_shap_prote('0_VS_2', top20_shap_prote)
    cols = ['FURIN', 'HGF', 'CD38', 'CCER2', 'GZMB', 'PCBD1', 'HNMT', 'IGSF3', 'IL15', 'GPA33', 'PRL', 'KLK14', 'BCHE', 'LRRN1', 'SELE', 'SERPINF1', 'TNFRSF1A', 'RBP5', 'EDIL3', 'CXCL14']
    print("0_VS_2:", cols)
    plot_bar(df_0_VS_2, f'Clust2_{exp}', cols, 18, 4, ['HC', 'Subtpye2'], 'Z-score')

    bar_biochem_plot(df, exp)
    PHQ_plot(df, exp)


def conti_pvalue(df, feature, group_col, group_order):
    """连续变量: Kruskal-Wallis 检验, 返回 p 值"""
    groups = [
        df.loc[df[group_col] == g, feature].dropna().to_numpy(dtype=float)
        for g in group_order
    ]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return float('nan')
    _, p = kruskalwallis(*groups)
    return p


def build_table_statistics(df, group_col, continue_cols):
    mean_df = df.groupby(group_col)[continue_cols].mean().T
    mean_df.columns = [f"{c}_mean" for c in mean_df.columns]

    std_df = df.groupby(group_col)[continue_cols].std().T
    std_df.columns = [f"{c}_std" for c in std_df.columns]

    # =========================
    # 4. 对每个蛋白计算 P 值
    # =========================
    pvals = {}

    for col in continue_cols:
        groups = [
            g[col].dropna().values
            for _, g in df.groupby(group_col)
        ]
        groups = [g for g in groups if len(g) > 1]  # 去掉样本数不足的组

        if len(groups) < 2:
            pvals[col] = np.nan
        elif len(groups) == 2:
            # 两组：Welch t-test；若想普通 t-test，改成 equal_var=True
            pvals[col] = stats.ttest_ind(groups[0], groups[1], equal_var=False).pvalue
        else:
            # 多组：单因素 ANOVA；若数据明显非正态，可改用 stats.kruskal(*groups).pvalue
            pvals[col] = stats.f_oneway(*groups).pvalue

    p_df = pd.DataFrame({"pvalue": pd.Series(pvals)})
    result = pd.concat([mean_df, std_df, p_df], axis=1)
    result.index.name = "cols"
    return result


def prote_biochem_statistics(exp, target_exp, continue_cols):
    df_NC, df_MDD = pd.read_csv(data_path + f"{exp}_NC.csv", low_memory=False), pd.read_csv(data_path + f"{exp}_MDD.csv", low_memory=False)
    df_NC['final_cluster'] = 0
    df_Y = pd.read_csv(result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(target_exp,"NMF", 2, 2), low_memory=False, usecols=['eid', 'final_cluster'])
    df_MDD = pd.merge(df_MDD, df_Y, how='inner', on='eid', sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)

    df = pd.concat([df_MDD, df_NC])
    df_1_VS_2 = df[df['final_cluster'].isin([1,2])]
    table = build_table_statistics(df_1_VS_2, 'final_cluster', continue_cols)
    out_file = os.path.join(result_path, f'Table_prote_biochem_{exp}_1_VS_2_clustering.xlsx')
    table.to_excel(out_file, index=True)

    df_0_VS_1 = df[df['final_cluster'].isin([0,1])]
    table = build_table_statistics(df_0_VS_1, 'final_cluster', continue_cols)
    out_file = os.path.join(result_path, f'Table_prote_biochem_{exp}_0_VS_1_clustering.xlsx')
    table.to_excel(out_file, index=True)

    df_0_VS_2 = df[df['final_cluster'].isin([0,2])]
    table = build_table_statistics(df_0_VS_2, 'final_cluster', continue_cols)
    out_file = os.path.join(result_path, f'Table_prote_biochem_{exp}_0_VS_2_clustering.xlsx')
    table.to_excel(out_file, index=True)

continue_cols = (cate17518_cols + ['PHQ_Total', 'PHQ'] + load_prote_sig_cols()+[c + '-0.0' for c in ['4609', '4620', '5375', '5386', '20127', '2050', '2060', '2070','2080']])


if __name__ == '__main__':
    # prote_plot('Diag_Discovery', 'train')
    # prote_plot('Diag_Test', 'test')

    # prote_biochem_statistics('Diag_Discovery', 'Diag_Discovery', continue_cols)
    # prote_biochem_statistics('Diag_Test', 'Diag_Test', continue_cols)

    prote_biochem_statistics('Diag_self_Discovery', 'Diag_Discovery', continue_cols)
    prote_biochem_statistics('Diag_self_Test', 'Diag_Test', continue_cols)
    prote_biochem_statistics('Diag_drug_Discovery', 'Diag_Discovery', continue_cols)
    prote_biochem_statistics('Diag_drug_Test', 'Diag_Test', continue_cols)


