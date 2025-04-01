from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, HDBSCAN, OPTICS, Birch, BisectingKMeans
from sklearn.decomposition import NMF, KernelPCA
from SUtils import *
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def k_means(df, W, n_clusters, algorithm):
    print("**********begin K means**********")
    clustering = KMeans(n_clusters=n_clusters, algorithm=algorithm, random_state=0, n_init="auto").fit(W)
    labels = clustering.labels_
    print(labels)
    print("**********end K means**********")
    silhouette, calinski_harabasz, davies_bouldin = silhouette_score(W, labels), calinski_harabasz_score(W, labels), davies_bouldin_score(W, labels)
    return labels, silhouette, calinski_harabasz, davies_bouldin


def bisecting_k_means(df, W, n_clusters, algorithm):
    print("**********begin Bisecting K means**********")
    clustering = BisectingKMeans(n_clusters=n_clusters, algorithm=algorithm, random_state=0).fit(W)
    labels = clustering.labels_
    print(labels)
    print("**********end Bisecting K means**********")
    silhouette, calinski_harabasz, davies_bouldin = silhouette_score(W, labels), calinski_harabasz_score(W, labels), davies_bouldin_score(W, labels)
    return labels, silhouette, calinski_harabasz, davies_bouldin


def spectral_clustering(df, W, n_clusters, eigen_solver):
    print("**********begin SpectralClustering**********")
    clustering = SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver, affinity='nearest_neighbors', assign_labels='cluster_qr', random_state=0).fit(W)
    labels = clustering.labels_
    print(labels)
    print("**********end SpectralClustering**********")
    silhouette, calinski_harabasz, davies_bouldin = silhouette_score(W, labels), calinski_harabasz_score(W, labels), davies_bouldin_score(W, labels)
    return labels, silhouette, calinski_harabasz, davies_bouldin


def h_clustering(df, W, n_clusters, linkage):
    print("**********begin AgglomerativeClustering**********")
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(W)
    labels = clustering.labels_
    print(labels)
    print("**********end AgglomerativeClustering**********")
    silhouette, calinski_harabasz, davies_bouldin = silhouette_score(W, labels), calinski_harabasz_score(W, labels), davies_bouldin_score(W, labels)
    return labels, silhouette, calinski_harabasz, davies_bouldin


def birch_cluster(df, W, n_clusters):
    print("**********begin Birch**********")
    brc = Birch(n_clusters=n_clusters, threshold=0.1)
    brc.fit(W)
    labels = brc.predict(W)
    print(labels)
    print("**********end Birch**********")
    silhouette, calinski_harabasz, davies_bouldin = silhouette_score(W, labels), calinski_harabasz_score(W, labels), davies_bouldin_score(W, labels)
    return labels, silhouette, calinski_harabasz, davies_bouldin


if __name__ == '__main__':
    dataset = "Diag_MDD"
    bernoulli_file = path + "result/t_test_result_bonferroni_Diag.csv"
    cols = pd.read_csv(bernoulli_file)['col']
    # cols = cols[:100]
    print("cols: {}".format(len(cols)))
    df_MDD = pd.read_csv(data_path + "{}.csv".format(dataset))
    # df_MDD = df_MDD.head(200)
    print("df_MDD Number: {}".format(df_MDD.shape[0]))
    df = df_MDD[cols].apply(lambda x: x - x.min(), axis=0)
    for algorithm_reducing in ['NMF', 'rbf_PCA', 'linear_PCA', 'poly_PCA', 'sigmoid_PCA', 'cosine_PCA']:
    # for algorithm_reducing in ['NMF']:
        if not os.path.exists(result_path + "{}/".format(algorithm_reducing)):
            os.makedirs(result_path + "{}/".format(algorithm_reducing))
        for n_components in range(2, 9):
            if(algorithm_reducing == 'NMF'):
                model = NMF(n_components=n_components, init='random', random_state=0)  # k为降维后的维度数
            elif(algorithm_reducing == 'rbf_PCA'):
                model = KernelPCA(n_components=n_components, kernel='rbf', random_state=0)
            elif (algorithm_reducing == 'linear_PCA'):
                model = KernelPCA(n_components=n_components, kernel='linear', random_state=0)
            elif (algorithm_reducing == 'poly_PCA'):
                model = KernelPCA(n_components=n_components, kernel='poly', random_state=0)
            elif (algorithm_reducing == 'sigmoid_PCA'):
                model = KernelPCA(n_components=n_components, kernel='sigmoid', random_state=0)
            elif (algorithm_reducing == 'cosine_PCA'):
                model = KernelPCA(n_components=n_components, kernel='cosine', random_state=0)
            W = model.fit_transform(df)

            f_silhouette = open(result_path + "{}/{}_components_silhouette_scores.csv".format(algorithm_reducing, n_components), "w")
            f_calinski_harabasz = open(result_path + "{}/{}_components_calinski_harabasz_scores.csv".format(algorithm_reducing, n_components), "w")
            f_DBI = open(result_path + "{}/{}_components_DBI_scores.csv".format(algorithm_reducing, n_components), "w")
            print("clust_num,kmeans_lloyd,bisect_kmeans_lloyd,spectral_arpack,hcluster_ward", file=f_silhouette)
            print("clust_num,kmeans_lloyd,bisect_kmeans_lloyd,spectral_arpack,hcluster_ward", file=f_calinski_harabasz)
            print("clust_num,kmeans_lloyd,bisect_kmeans_lloyd,spectral_arpack,hcluster_ward", file=f_DBI)
            for cluster_num in range(2, 9):
                print("----------------------------------algorithm_reducing:{}, n_components:{}, cluster_num:{}----------------------------------".format(algorithm_reducing, n_components, cluster_num))
                kmeans_lloyd_labels, kmeans_lloyd_silhouette, kmeans_lloyd_calinski_harabasz, kmeans_lloyd_DBI = k_means(df, W, cluster_num, 'lloyd')
                # kmeans_elkan_labels, kmeans_elkan_silhouette, kmeans_elkan_calinski_harabasz, kmeans_elkan_DBI = k_means(df, W, cluster_num, 'elkan')
                bisect_kmeans_lloyd_labels, bisect_kmeans_lloyd_silhouette, bisect_kmeans_lloyd_calinski_harabasz, bisect_kmeans_lloyd_DBI = bisecting_k_means(df, W, cluster_num, 'lloyd')
                # bisect_kmeans_elkan_labels, bisect_kmeans_elkan_silhouette, bisect_kmeans_elkan_calinski_harabasz, bisect_kmeans_elkan_DBI = bisecting_k_means(df, W, cluster_num, 'elkan')
                spectral_arpack_labels, spectral_arpack_silhouette, spectral_arpack_calinski_harabasz, spectral_arpack_DBI = spectral_clustering(df, W, cluster_num, 'arpack')
                # spectral_lobpcg_labels, spectral_lobpcg_silhouette, spectral_lobpcg_calinski_harabasz, spectral_lobpcg_DBI = spectral_clustering(df, W, cluster_num, 'lobpcg')
                # spectral_amg_labels, spectral_amg_silhouette, spectral_amg_calinski_harabasz, spectral_amg_DBI = spectral_clustering(df, W, cluster_num, 'amg')
                hcluster_ward_labels, hcluster_ward_silhouette, hcluster_ward_calinski_harabasz, hcluster_ward_DBI = h_clustering(df, W, cluster_num, 'ward')
                # hcluster_complete_labels, hcluster_complete_silhouette, hcluster_complete_calinski_harabasz, hcluster_complete_DBI = h_clustering(df, W, cluster_num, 'complete')
                # birch_labels, birch_silhouette, birch_calinski_harabasz = birch_cluster(df, W, cluster_num)
                print("{},{},{},{},{}".format(cluster_num, kmeans_lloyd_silhouette, bisect_kmeans_lloyd_silhouette,
                                                    spectral_arpack_silhouette, hcluster_ward_silhouette), file=f_silhouette)
                print("{},{},{},{},{}".format(cluster_num, kmeans_lloyd_calinski_harabasz,
                                                    bisect_kmeans_lloyd_calinski_harabasz,
                                                    spectral_arpack_calinski_harabasz,
                                                    hcluster_ward_calinski_harabasz), file=f_calinski_harabasz)
                print("{},{},{},{},{}".format(cluster_num, kmeans_lloyd_DBI,
                                                             bisect_kmeans_lloyd_DBI,
                                                             spectral_arpack_DBI,
                                                             hcluster_ward_DBI), file=f_DBI)
                rst_df = pd.DataFrame()
                rst_df['eid'] = df_MDD['eid']
                rst_df['kmeans_lloyd'] = kmeans_lloyd_labels
                # rst_df['kmeans_elkan'] = kmeans_elkan_labels
                rst_df['bisect_kmeans_lloyd'] = bisect_kmeans_lloyd_labels
                # rst_df['bisect_kmeans_elkan'] = bisect_kmeans_elkan_labels
                rst_df['spectral_arpack'] = spectral_arpack_labels
                # rst_df['spectral_lobpcg'] = spectral_lobpcg_labels
                # rst_df['spectral_amg'] = spectral_amg_labels
                rst_df['hcluster_ward'] = hcluster_ward_labels
                # rst_df['hcluster_complete'] = hcluster_complete_labels
                rst_file = result_path + "{}/{}_components_cluster_rst_{}.csv".format(algorithm_reducing, n_components, cluster_num)
                rst_df.to_csv(rst_file, index=None)





