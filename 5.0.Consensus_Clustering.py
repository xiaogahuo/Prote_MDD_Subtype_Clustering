import numpy as np
from SUtils import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# # 示例
# labels = ['A', 'B', 'A', 'C', 'B']
# similarity_matrix = compute_similarity_matrix(labels)
# print(similarity_matrix)
def perform_consensus(exp):
    for n_components in range(2, 9):
        for cluster_num in range(2, 9):
            # 基于4个聚类算法的聚类结果生成加权similarity_matrix
            rst_file = result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(exp, "NMF", n_components, cluster_num)
            df = pd.read_csv(rst_file)
            cluster_algorithms = df.columns.to_list()[1:4]
            n_algorithms = len(cluster_algorithms)
            m_samples = df.shape[0]
            print(cluster_algorithms, n_algorithms, m_samples)
            similarity_matrix = np.zeros((m_samples, m_samples))
            for col in cluster_algorithms:
                print(col)
                similarity_matrix += compute_similarity_matrix(df[col])
            similarity_matrix = similarity_matrix/n_algorithms

            # 将相似矩阵转换为距离矩阵
            distance_matrix = 1 - similarity_matrix
            # 使用Ward方法进行层次聚类
            linked = linkage(squareform(distance_matrix), method='ward')
            # 根据cluster_num进行聚类
            clusters = fcluster(linked, cluster_num, criterion='maxclust')
            print("聚类结果：", clusters)
            # 保存final cluster结果到 rst_file
            df['final_cluster'] = pd.Series(clusters)
            df.to_csv(rst_file, index=False)


if __name__ == '__main__':
    perform_consensus('Diag_Discovery')
    perform_consensus('Diag_Test')

    exp = 'Diag_Discovery'
    for n_components in range(2, 9):
        for cluster_num in range(2, 9):
            # 基于3个聚类算法的聚类结果生成加权similarity_matrix
            rst_file = result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(exp, "NMF", n_components, cluster_num)
            df = pd.read_csv(rst_file)
            df['final_cluster'] = df['final_cluster'].replace({1: 2, 2: 1})
            df.to_csv(rst_file, index=False)
    exp = 'Diag_Test'
    for n_components in range(2, 9):
        for cluster_num in range(2, 9):
            # 基于3个聚类算法的聚类结果生成加权similarity_matrix
            rst_file = result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(exp, "NMF", n_components, cluster_num)
            df = pd.read_csv(rst_file)
            df['final_cluster'] = df['final_cluster'].replace({1: 2, 2: 1})
            df.to_csv(rst_file, index=False)




