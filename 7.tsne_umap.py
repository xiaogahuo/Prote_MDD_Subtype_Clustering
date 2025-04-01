import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import pandas as pd
from SUtils import *


dataset = "Diag_MDD"
bernoulli_file = path + "result/t_test_result_bonferroni_Diag.csv"
cols = pd.read_csv(bernoulli_file)['col']
df_MDD = pd.read_csv(data_path + "{}.csv".format(dataset))
print(df_MDD.head(20))
X = df_MDD[cols]
for n_components in range(2, 9):
    for cluster_num in range(2, 9):
        # 加载数据集
        rst_file = result_path + "{}/{}_components_cluster_rst_{}.csv".format("NMF", n_components, cluster_num)
        img_file = img_path + "{}/{}_components_cluster_tsne_umap_{}.png".format("NMF", n_components, cluster_num)
        df_clst_rst = pd.read_csv(rst_file)
        y = df_clst_rst['final_cluster']

        # UMAP降维
        umap_reducer = umap.UMAP(random_state=42)
        X_umap = umap_reducer.fit_transform(X)

        # t-SNE降维
        tsne_reducer = TSNE(random_state=42)
        X_tsne = tsne_reducer.fit_transform(X)

        # 绘制UMAP结果
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='Spectral', s=5)
        plt.title('UMAP')

        # 绘制t-SNE结果
        plt.subplot(122)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='Spectral', s=5)
        plt.title('t-SNE')

        plt.savefig(img_file)
        plt.clf()
