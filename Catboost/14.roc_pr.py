from SUtils import *
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def get_full_eval(exp, dataset):
    """在 final model 上 使用 Discovery 集进行预测"""
    df = pd.read_csv(result_path + "{}_{}/{}_{}_full_eval.csv".format(exp, dataset, exp, dataset))
    return df['label'].values, df['pred_prob'].values


def get_test_eval(exp, dataset):
    """最终锁定模型在 Test 集上的预测"""
    df = pd.read_csv(result_path + "{}_{}/{}_{}_test_eval.csv".format(exp, dataset, exp, dataset))
    return df['label'].values, df['pred_prob'].values


def get_fold_valid_evals(exp, dataset):
    """最优超参下的 5 折验证集预测"""
    labels, probs = [], []
    for fold in range(5):
        df = pd.read_csv(result_path + "{}_{}/fold{}_valid_eval.csv".format(exp, dataset, fold))
        labels.append(df['label'].values)
        probs.append(df['pred_prob'].values)
    return labels, probs


def get_fold_eval(exp, dataset, fold, d_type):
    """读取某折训练集(train)/验证集(valid)的预测: 返回 (pred_prob, label)"""
    df = pd.read_csv(result_path + "{}_{}/fold{}_{}_eval.csv".format(exp, dataset, fold, d_type))
    return df['pred_prob'].values, df['label'].values


def ensure_img_dir(exp, dataset):
    d = img_path + "{}_{}/".format(exp, dataset)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


# ==================== 原有: Test ROC / Test PR ====================
def plot_final_model_and_test_roc_pr(exp, dataset):
    # full_labels, full_probs = get_full_eval(exp, dataset)
    # full_fpr, full_tpr, _ = roc_curve(full_labels, full_probs, pos_label=1)
    # full_AUC = auc(full_fpr, full_tpr)
    # full_precision, full_recall, _ = precision_recall_curve(full_labels, full_probs, pos_label=1)
    # full_AP = average_precision_score(full_labels, full_probs, pos_label=1)

    labels, probs = get_test_eval(exp, dataset)
    fpr, tpr, _ = roc_curve(labels, probs, pos_label=1)
    AUC = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(labels, probs, pos_label=1)
    AP = average_precision_score(labels, probs, pos_label=1)

    plt.figure(figsize=(14, 6), dpi=720)
    plt.subplot(1, 2, 1)  # ROC
    # plt.plot(full_fpr, full_tpr, color='#222222', lw=2, label='ROC on Final Model (AUC={:.3f})'.format(full_AUC))
    plt.plot(fpr, tpr, color='r', lw=2, label='Test ROC on Final Model (AUC={:.3f})'.format(AUC))
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)  # PR
    # plt.plot(full_recall, full_precision, color='#222222', lw=2, label='PR on Final Model (AUPRC={:.3f})'.format(full_AP))
    plt.plot(recall, precision, color='r', lw=2, label='Test PR on Final Model (AUPRC={:.3f})'.format(AP))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR')
    plt.legend(loc='upper right')

    d = ensure_img_dir(exp, dataset)
    plt.savefig(d + "{}_{}_Test_ROC_PR.png".format(exp, dataset), bbox_inches='tight')
    plt.clf()
    plt.close()


# ==================== 新增 图1: 5折融合总图(训练集+验证集 ROC+PR, AUC/AUPRC均值±SD) ====================
def plot_cv_fusion(exp, dataset):
    lw = 1
    text_size = 18
    types = ['train', 'valid']
    colors = ['#222222', 'g']
    plt.figure(figsize=(14, 6), dpi=720)

    plt.subplot(1, 2, 1)  # ROC
    mean_fpr = np.linspace(0, 1, 100)
    for d_type, color in zip(types, colors):
        tprs, aucs = [], []
        for fold in range(5):
            probs, labels = get_fold_eval(exp, dataset, fold, d_type)
            fpr, tpr, _ = roc_curve(labels, probs, pos_label=1)
            aucs.append(auc(fpr, tpr))
            plt.plot(fpr, tpr, lw=lw / 2, alpha=0.15, color=color)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc, std_auc = np.mean(aucs), np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color=color, lw=2, alpha=.8,
                 label='{} AUC={:.3f} ± {:.3f}'.format(d_type, mean_auc, std_auc))
        std_tpr = np.std(tprs, axis=0)
        plt.fill_between(mean_fpr, np.minimum(mean_tpr + std_tpr, 1), np.maximum(mean_tpr - std_tpr, 0), color=color, alpha=.2)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('False Positive Rate', fontsize=text_size, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=text_size, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)  # PR
    mean_rc = np.linspace(0, 1, 100)
    for d_type, color in zip(types, colors):
        prs, ap_s = [], []
        for fold in range(5):
            probs, labels = get_fold_eval(exp, dataset, fold, d_type)
            pr, rc, _ = precision_recall_curve(labels, probs, pos_label=1)
            pr, rc = pr[::-1], rc[::-1]
            ap_s.append(average_precision_score(labels, probs, pos_label=1))
            plt.plot(rc, pr, lw=lw / 2, alpha=0.15, color=color)
            interp_pr = np.interp(mean_rc, rc, pr)
            prs.append(interp_pr)
        mean_pr = np.mean(prs, axis=0)
        plt.plot(mean_rc, mean_pr, color=color, lw=2, alpha=.8,
                 label='{} AUPRC={:.3f} ± {:.3f}'.format(d_type, np.mean(ap_s), np.std(ap_s)))
        std_pr = np.std(prs, axis=0)
        plt.fill_between(mean_rc, np.minimum(mean_pr + std_pr, 1), np.maximum(mean_pr - std_pr, 0), color=color, alpha=.2)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Recall', fontsize=text_size, fontweight='bold')
    plt.ylabel('Precision', fontsize=text_size, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(loc='upper right')

    d = ensure_img_dir(exp, dataset)
    plt.savefig(d + "{}_{}_CV_fusion_ROC_PR.png".format(exp, dataset), bbox_inches='tight')
    plt.clf()
    plt.close()


def generate_dataset(exp, dataset):
    # 新增: 图1 5折融合总图 + 图2-6 单折图
    plot_cv_fusion(exp, dataset)
    plot_final_model_and_test_roc_pr(exp, dataset)


if __name__ == '__main__':
    # generate_dataset('Diag_Discovery', '0_VS_12')
    # generate_dataset('Diag_Discovery', '0_VS_1')
    # generate_dataset('Diag_Discovery', '0_VS_2')

    generate_dataset('Diag_self_Discovery', '0_VS_12')
    generate_dataset('Diag_self_Discovery', '0_VS_1')
    generate_dataset('Diag_self_Discovery', '0_VS_2')

    generate_dataset('Diag_drug_Discovery', '0_VS_12')
    generate_dataset('Diag_drug_Discovery', '0_VS_1')
    generate_dataset('Diag_drug_Discovery', '0_VS_2')
