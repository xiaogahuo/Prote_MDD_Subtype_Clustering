import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import umap
import pandas as pd
from SUtils import *


def display_tsne_umap(exp):
    bernoulli_file = result_path + f"t_test_result_bonferroni_{exp}.csv"
    cols = pd.read_csv(bernoulli_file)['col']
    df_MDD = pd.read_csv(data_path + f"{exp}_MDD.csv")
    print(df_MDD.head(20))
    X = df_MDD[cols]
    # cmap_colors = ListedColormap(['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#B39FDB', '#0077BB', '#EE7733', '#33BBEE'])
    for n_components in range(2, 3):
        for cluster_num in range(2, 3):
            # 加载数据集
            rst_file = result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(exp, "NMF", n_components, cluster_num)
            img_file = img_path + "{}/{}/{}_components_cluster_tsne_umap_{}.png".format(exp, "NMF", n_components, cluster_num)
            df_clst_rst = pd.read_csv(rst_file)
            y = df_clst_rst['final_cluster']

            # UMAP降维
            umap_reducer = umap.UMAP(random_state=0)
            X_umap = umap_reducer.fit_transform(X)

            # t-SNE降维
            tsne_reducer = TSNE(random_state=0)
            X_tsne = tsne_reducer.fit_transform(X)

            # 绘制UMAP结果
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="vlag", s=6)
            plt.title('UMAP')

            # 绘制t-SNE结果
            plt.subplot(122)
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="vlag", s=6)
            plt.title('t-SNE')

            plt.savefig(img_file)
            plt.clf()


if __name__ == '__main__':
    display_tsne_umap('Diag')
    display_tsne_umap('DiagA')
    display_tsne_umap('DiagB')