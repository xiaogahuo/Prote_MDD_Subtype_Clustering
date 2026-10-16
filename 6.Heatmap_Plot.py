import numpy as np
import pandas as pd

from SUtils import *
import seaborn as sns
import matplotlib.pyplot as plt


def plot_bonferroni_mean(exp):
    cols = load_prote_sig_cols()
    df_MDD, df_NC = pd.read_csv(data_path + f"{exp}_MDD.csv", low_memory=False,
                                usecols=['eid'] + cols), pd.read_csv(data_path + f"{exp}_NC.csv",
                                                                               low_memory=False,
                                                                               usecols=['eid'] + cols)
    df_NC['final_cluster'] = 0
    df_Y = pd.read_csv(result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(exp, "NMF", 2, 2), low_memory=False,
                       usecols=['eid', 'final_cluster'])
    df_MDD = pd.merge(df_MDD, df_Y, on='eid', how='inner')
    df_MDD_clst1, df_MDD_clst2 = df_MDD[df_MDD['final_cluster'] == 1], df_MDD[df_MDD['final_cluster'] == 2]
    bonferroni_mean_file = open(result_path + f"bonferroni_mean_zscore_{exp}.csv", "w")
    print("Prote_Name,HC,clust1,clust2", file=bonferroni_mean_file)
    for prote, i in zip(cols, range(len(cols))):
        print('i:{}, prote:{}'.format(i, prote))
        m_NC, m_clst1, m_clst2 = df_NC[prote].mean(), df_MDD_clst1[prote].mean(), df_MDD_clst2[prote].mean()
        print("{},{},{},{}".format(prote, m_NC, m_clst1, m_clst2), file=bonferroni_mean_file)

    df = pd.read_csv(result_path + f"bonferroni_mean_zscore_{exp}.csv", usecols=['HC', 'clust1', 'clust2'])
    plt.figure(figsize=(6, 24))
    myPlot = sns.heatmap(df, annot=False, yticklabels=False, cmap="vlag", center=0)
    myPlot.xaxis.tick_top()
    plt.savefig(img_path + f"bonferroni_mean_zscore_{exp}.png")


if __name__ == '__main__':
    plot_bonferroni_mean('Diag_Discovery')
    plot_bonferroni_mean('Diag_Test')