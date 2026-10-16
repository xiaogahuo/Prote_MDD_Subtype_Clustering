import shap
import matplotlib.pyplot as plt
from SUtils import *


def get_topN_reason(exp, dataset, feature_importance, cols, d_type, top_num=500):
    # 输出特征重要性值(mean|SHAP|)最高的N个特征
    feature_importance_dict = dict(zip(cols, feature_importance))
    new_dict = dict(sorted(feature_importance_dict.items(), key=lambda e: e[1], reverse=True))
    f = open(img_path + "{}_{}/{}_{}_{}_SHAP.csv".format(exp, dataset, exp, dataset, d_type), "w")
    print("key,value", file=f)
    for k, v in new_dict.items():
        if top_num > 0:
            print("{},{}".format(k, v), file=f)
            top_num -= 1
        else:
            break


def plot_shap(exp, dataset, d_type):
    cols = load_prote_sig_cols()
    base = data_path + "cross_valid/{}_{}/".format(exp, dataset)
    if d_type == 'train':
        X = pd.read_csv(base + "full_train.csv")[cols]   # Discovery 全集
        shap_values = np.load(result_path + "{}_{}/{}_full_shap.npy".format(exp, dataset, dataset))
    else:
        X = pd.read_csv(base + "test.csv")[cols]          # Test 集
        shap_values = np.load(result_path + "{}_{}/{}_test_shap.npy".format(exp, dataset, dataset))
    print("exp: {}_{}: X_{} shape: {}, shap_values: {}".format(exp, dataset, d_type, X.shape, shap_values.shape))

    plt.figure(figsize=(20, 8), dpi=720)
    plt.subplot(1, 2, 1)
    shap.plots.violin(shap_values, features=X, feature_names=cols, plot_type="layered_violin", show=False, max_display=20)
    plt.xlabel('SHAP value')
    plt.subplot(1, 2, 2)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.xlabel('Mean(|SHAP|)')
    out_dir = img_path + "{}_{}/".format(exp, dataset)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(out_dir + "{}_{}_{}_SHAP.png".format(exp, dataset, d_type), dpi=720)
    plt.clf()  # CLEAR FIGURE

    # Feature 重要性导出
    feature_importance = np.absolute(shap_values).mean(axis=0)
    get_topN_reason(exp, dataset, feature_importance, cols, d_type, top_num=len(cols))


if __name__ == '__main__':
    # plot_shap('Diag_Discovery', '0_VS_12', 'train')
    # plot_shap('Diag_Discovery', '0_VS_12', 'test')
    # plot_shap('Diag_Discovery', '0_VS_1', 'train')
    # plot_shap('Diag_Discovery', '0_VS_1', 'test')
    # plot_shap('Diag_Discovery', '0_VS_2', 'train')
    # plot_shap('Diag_Discovery', '0_VS_2', 'test')

    plot_shap('Diag_self_Discovery', '0_VS_12', 'train')
    plot_shap('Diag_self_Discovery', '0_VS_12', 'test')
    plot_shap('Diag_self_Discovery', '0_VS_1', 'train')
    plot_shap('Diag_self_Discovery', '0_VS_1', 'test')
    plot_shap('Diag_self_Discovery', '0_VS_2', 'train')
    plot_shap('Diag_self_Discovery', '0_VS_2', 'test')

    plot_shap('Diag_drug_Discovery', '0_VS_12', 'train')
    plot_shap('Diag_drug_Discovery', '0_VS_12', 'test')
    plot_shap('Diag_drug_Discovery', '0_VS_1', 'train')
    plot_shap('Diag_drug_Discovery', '0_VS_1', 'test')
    plot_shap('Diag_drug_Discovery', '0_VS_2', 'train')
    plot_shap('Diag_drug_Discovery', '0_VS_2', 'test')
