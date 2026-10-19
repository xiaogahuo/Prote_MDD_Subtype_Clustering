from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, HDBSCAN, OPTICS, Birch, BisectingKMeans
from sklearn.decomposition import NMF, KernelPCA
from SUtils import *
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
from scipy.spatial.distance import cdist


def k_means(W, n_clusters, algorithm):
    print("**********begin K means**********")
    clustering = KMeans(n_clusters=n_clusters, algorithm=algorithm, random_state=0, n_init="auto").fit(W)
    labels = clustering.labels_
    print(labels)
    print("**********end K means**********")
    silhouette, calinski_harabasz, davies_bouldin = silhouette_score(W, labels), calinski_harabasz_score(W, labels), davies_bouldin_score(W, labels)
    return labels, silhouette, calinski_harabasz, davies_bouldin, clustering


def bisecting_k_means(W, n_clusters, algorithm):
    print("**********begin Bisecting K means**********")
    clustering = BisectingKMeans(n_clusters=n_clusters, algorithm=algorithm, random_state=0).fit(W)
    labels = clustering.labels_
    print(labels)
    print("**********end Bisecting K means**********")
    silhouette, calinski_harabasz, davies_bouldin = silhouette_score(W, labels), calinski_harabasz_score(W, labels), davies_bouldin_score(W, labels)
    return labels, silhouette, calinski_harabasz, davies_bouldin


def spectral_clustering(W, n_clusters, eigen_solver):
    print("**********begin SpectralClustering**********")
    clustering = SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver, affinity='nearest_neighbors', assign_labels='cluster_qr', random_state=0).fit(W)
    labels = clustering.labels_
    print(labels)
    print("**********end SpectralClustering**********")
    silhouette, calinski_harabasz, davies_bouldin = silhouette_score(W, labels), calinski_harabasz_score(W, labels), davies_bouldin_score(W, labels)
    return labels, silhouette, calinski_harabasz, davies_bouldin


def h_clustering(W, n_clusters, linkage):
    print("**********begin AgglomerativeClustering**********")
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(W)
    labels = clustering.labels_
    print(labels)
    print("**********end AgglomerativeClustering**********")
    silhouette, calinski_harabasz, davies_bouldin = silhouette_score(W, labels), calinski_harabasz_score(W, labels), davies_bouldin_score(W, labels)
    return labels, silhouette, calinski_harabasz, davies_bouldin


def birch_cluster(W, n_clusters):
    print("**********begin Birch**********")
    brc = Birch(n_clusters=n_clusters, threshold=0.1)
    brc.fit(W)
    labels = brc.predict(W)
    print(labels)
    print("**********end Birch**********")
    silhouette, calinski_harabasz, davies_bouldin = silhouette_score(W, labels), calinski_harabasz_score(W, labels), davies_bouldin_score(W, labels)
    return labels, silhouette, calinski_harabasz, davies_bouldin


def clustering_perform(exp_train):
    cols = load_prote_sig_cols()
    print("cols: {}".format(len(cols)))
    df_MDD = pd.read_csv(data_path + f"{exp_train}_MDD.csv")
    X_original = df_MDD[cols]
    col_min = X_original.min(axis=0)
    X_NMF = X_original - col_min
    print("df_MDD Number: {}".format(df_MDD.shape[0]))
    for algorithm_reducing in ['NMF', 'rbf', 'linear', 'poly', 'sigmoid', 'cosine']:
        os.makedirs(result_path + f"{exp_train}/{algorithm_reducing}/", exist_ok=True)
        silhouette_results, calinski_harabasz_results, dbi_results = [], [], []
        for n_components in range(2, 9):
            if (algorithm_reducing == 'NMF'):
                model = NMF(n_components=n_components, init='random', random_state=0)
                X = X_NMF
            else:
                model = KernelPCA(n_components=n_components, kernel=algorithm_reducing, random_state=0)
                X = X_original
            W = model.fit_transform(X)
            joblib.dump(model, result_path + f"{exp_train}/{algorithm_reducing}/{n_components}_dimensionality reduction.joblib")

            for cluster_num in range(2, 9):
                print("----------------------------------algorithm_reducing:{}, n_components:{}, cluster_num:{}----------------------------------".format(algorithm_reducing, n_components, cluster_num))
                kmeans_lloyd_labels, kmeans_lloyd_silhouette, kmeans_lloyd_calinski_harabasz, kmeans_lloyd_DBI, kmeans_clustering_model = k_means(W, cluster_num, 'lloyd')
                spectral_arpack_labels, spectral_arpack_silhouette, spectral_arpack_calinski_harabasz, spectral_arpack_DBI = spectral_clustering(W, cluster_num, 'arpack')
                hcluster_ward_labels, hcluster_ward_silhouette, hcluster_ward_calinski_harabasz, hcluster_ward_DBI = h_clustering(W, cluster_num, 'ward')

                silhouette_results.append({
                    "components": n_components,
                    "clust_num": cluster_num,
                    "kmeans_lloyd": kmeans_lloyd_silhouette,
                    "spectral_arpack": spectral_arpack_silhouette,
                    "hcluster_ward": hcluster_ward_silhouette
                })
                calinski_harabasz_results.append({
                    "components": n_components,
                    "clust_num": cluster_num,
                    "kmeans_lloyd": kmeans_lloyd_calinski_harabasz,
                    "spectral_arpack": spectral_arpack_calinski_harabasz,
                    "hcluster_ward": hcluster_ward_calinski_harabasz
                })
                dbi_results.append({
                    "components": n_components,
                    "clust_num": cluster_num,
                    "kmeans_lloyd": kmeans_lloyd_DBI,
                    "spectral_arpack": spectral_arpack_DBI,
                    "hcluster_ward": hcluster_ward_DBI
                })

                rst_df = pd.DataFrame()
                rst_df['eid'] = df_MDD['eid']
                rst_df['kmeans_lloyd'] = kmeans_lloyd_labels
                rst_df['spectral_arpack'] = spectral_arpack_labels
                rst_df['hcluster_ward'] = hcluster_ward_labels
                rst_file = result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(exp_train, algorithm_reducing, n_components, cluster_num)
                rst_df.to_csv(rst_file, index=None)
                joblib.dump(kmeans_clustering_model, result_path + f"{exp_train}/{algorithm_reducing}/{n_components}_components_clustering{cluster_num}_kmeans_lloyd.joblib")

                # 谱/层次聚类无 predict 方法, 保存各簇在 NMF 空间中的质心, 供 Test 用最近质心法打标签
                for name, labels in [('spectral_arpack', spectral_arpack_labels), ('hcluster_ward', hcluster_ward_labels)]:
                    centroids = np.array([W[labels == c].mean(axis=0) for c in range(cluster_num)])
                    np.save(result_path + f"{exp_train}/{algorithm_reducing}/{n_components}_components_clustering_{name}_{cluster_num}_centroids.npy", centroids)

        f_silhouette = pd.DataFrame(silhouette_results)
        f_calinski_harabasz = pd.DataFrame(calinski_harabasz_results)
        f_DBI = pd.DataFrame(dbi_results)
        excel_path = result_path + f"{exp_train}/{algorithm_reducing}/Clustering_SI_CH_DBI.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            f_silhouette.to_excel(writer, sheet_name="Silhouette", index=False)
            f_calinski_harabasz.to_excel( writer, sheet_name="Calinski_Harabasz", index=False)
            f_DBI.to_excel(writer, sheet_name="DBI",index=False)


def predict_test(exp_train='Diag_Discovery', exp_test='Diag_Test'):
    print("**********begin predict Test labels with Discovery models**********")
    cols = load_prote_sig_cols()
    print("cols: {}".format(len(cols)))
    df_MDD = pd.read_csv(data_path + "{}_MDD.csv".format(exp_test))
    X_original = df_MDD[cols]
    col_min = X_original.min(axis=0)
    X_NMF = X_original - col_min
    for algorithm_reducing in ['NMF', 'rbf', 'linear', 'poly', 'sigmoid', 'cosine']:
        os.makedirs(result_path + f"{exp_test}/{algorithm_reducing}/", exist_ok=True)
        silhouette_results, calinski_harabasz_results, dbi_results = [], [], []
        for n_components in range(2, 9):
            if (algorithm_reducing == 'NMF'):
                X = X_NMF
            else:
                X = X_original
            model = joblib.load(result_path + f"{exp_train}/{algorithm_reducing}/{n_components}_dimensionality reduction.joblib")
            W = model.transform(X)  # must be transform, not fit_transform

            for cluster_num in range(2, 9):
                kmeans = joblib.load(result_path + f"{exp_train}/{algorithm_reducing}/{n_components}_components_clustering{cluster_num}_kmeans_lloyd.joblib")
                kmeans_labels = kmeans.predict(W)  # KMeans 原生 predict
                centroids_spectral = np.load(result_path + f"{exp_train}/{algorithm_reducing}/{n_components}_components_clustering_spectral_arpack_{cluster_num}_centroids.npy")
                centroids_hcluster = np.load(result_path + f"{exp_train}/{algorithm_reducing}/{n_components}_components_clustering_hcluster_ward_{cluster_num}_centroids.npy")
                spectral_labels = cdist(W, centroids_spectral).argmin(axis=1)  # 最近质心
                hcluster_labels = cdist(W, centroids_hcluster).argmin(axis=1)  # 最近质心

                kmeans_lloyd_silhouette, kmeans_lloyd_calinski_harabasz, kmeans_lloyd_DBI = silhouette_score(W, kmeans_labels), calinski_harabasz_score(W, kmeans_labels), davies_bouldin_score(W, kmeans_labels)
                spectral_arpack_silhouette, spectral_arpack_calinski_harabasz, spectral_arpack_DBI = silhouette_score(W, spectral_labels), calinski_harabasz_score(W, spectral_labels), davies_bouldin_score(W, spectral_labels)
                hcluster_ward_silhouette, hcluster_ward_calinski_harabasz, hcluster_ward_DBI = silhouette_score(W, hcluster_labels), calinski_harabasz_score(W, hcluster_labels), davies_bouldin_score(W, hcluster_labels)
                silhouette_results.append({
                    "components": n_components,
                    "clust_num": cluster_num,
                    "kmeans_lloyd": kmeans_lloyd_silhouette,
                    "spectral_arpack": spectral_arpack_silhouette,
                    "hcluster_ward": hcluster_ward_silhouette
                })
                calinski_harabasz_results.append({
                    "components": n_components,
                    "clust_num": cluster_num,
                    "kmeans_lloyd": kmeans_lloyd_calinski_harabasz,
                    "spectral_arpack": spectral_arpack_calinski_harabasz,
                    "hcluster_ward": hcluster_ward_calinski_harabasz
                })
                dbi_results.append({
                    "components": n_components,
                    "clust_num": cluster_num,
                    "kmeans_lloyd": kmeans_lloyd_DBI,
                    "spectral_arpack": spectral_arpack_DBI,
                    "hcluster_ward": hcluster_ward_DBI
                })

                rst_df = pd.DataFrame()
                rst_df['eid'] = df_MDD['eid']
                rst_df['kmeans_lloyd'] = kmeans_labels
                rst_df['spectral_arpack'] = spectral_labels
                rst_df['hcluster_ward'] = hcluster_labels
                rst_file = result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(exp_test, algorithm_reducing,
                                                                                         n_components, cluster_num)
                rst_df.to_csv(rst_file, index=None)

        f_silhouette = pd.DataFrame(silhouette_results)
        f_calinski_harabasz = pd.DataFrame(calinski_harabasz_results)
        f_DBI = pd.DataFrame(dbi_results)
        excel_path = result_path + f"{exp_test}/{algorithm_reducing}/Clustering_SI_CH_DBI.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            f_silhouette.to_excel(writer, sheet_name="Silhouette", index=False)
            f_calinski_harabasz.to_excel(writer, sheet_name="Calinski_Harabasz", index=False)
            f_DBI.to_excel(writer, sheet_name="DBI", index=False)


if __name__ == '__main__':
    clustering_perform('Diag_Discovery')
    predict_test('Diag_Discovery', 'Diag_Test')
