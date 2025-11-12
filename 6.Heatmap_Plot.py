import numpy as np
import pandas as pd

from SUtils import *
import seaborn as sns
import matplotlib.pyplot as plt


def consensus_cluster_heatmap(exp):
    if not os.path.exists(img_path + "{}/{}/".format(exp, "NMF")):
        os.makedirs(img_path + "{}/{}/".format(exp, "NMF"))

    for n_components in range(2, 3):
        for cluster_num in range(2, 3):
            # 按final_cluster对聚类结果排序，基于3个聚类算法的聚类结果生成加权similarity_matrix
            rst_file = result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(exp, "NMF", n_components, cluster_num)
            img_file = img_path + "{}/{}/{}_components_cluster_rst_{}.png".format(exp, "NMF", n_components, cluster_num)
            df = pd.read_csv(rst_file)
            df = df.sort_values(by='final_cluster')
            cluster_algorithms = df.columns.to_list()[1:4]
            n_algorithms = len(cluster_algorithms)
            m_samples = df.shape[0]
            print(cluster_algorithms, n_algorithms, m_samples)
            similarity_matrix = np.zeros((m_samples, m_samples))
            for col in cluster_algorithms:
                print(col)
                similarity_matrix += compute_similarity_matrix(df[col])
            similarity_matrix = similarity_matrix/n_algorithms

            sns.heatmap(similarity_matrix, annot=False, square=True)
            plt.savefig(img_file)
            plt.clf()


def plot_bonferroni_mean(exp):
    bernoulli_file = result_path + f"t_test_result_bonferroni_{exp}.csv"
    cols = pd.read_csv(bernoulli_file)['col'].tolist()
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
    print("Prote_Name,HC,MDD_clust2,MDD_clust1", file=bonferroni_mean_file)
    for prote, i in zip(cols, range(len(cols))):
        print('i:{}, prote:{}'.format(i, prote))
        mean_NC, mean_clst1, mean_clst2 = df_NC[prote].mean(), df_MDD_clst1[prote].mean(), df_MDD_clst2[prote].mean()
        print("{},{},{},{}".format(prote, mean_NC, mean_clst2, mean_clst1), file=bonferroni_mean_file)

    df = pd.read_csv(result_path + f"bonferroni_mean_zscore_{exp}.csv", usecols=['HC', 'MDD_clust2', 'MDD_clust1'])
    plt.figure(figsize=(6, 24))
    myPlot = sns.heatmap(df, annot=False, yticklabels=False, cmap="vlag", center=0)
    myPlot.xaxis.tick_top()
    plt.savefig(img_path + f"bonferroni_mean_zscore_{exp}.png")


if __name__ == '__main__':
    consensus_cluster_heatmap('Diag')
    consensus_cluster_heatmap('DiagA')
    consensus_cluster_heatmap('DiagB')
    plot_bonferroni_mean('Diag')
    plot_bonferroni_mean('DiagA')
    plot_bonferroni_mean('DiagB')