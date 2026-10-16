from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from SUtils import *
import shap
import json
import itertools

# ================= 超参数搜索配置 =================
PARAM_GRID = list(itertools.product(
    [0.01, 0.02, 0.05],   # learning_rate
    [4, 6],               # max_depth
    [0.5, 0.8]            # subsample
))
N_FOLDS = 5
N_ESTIMATORS_CV = 200     # 5折CV时的迭代上限(配合 early stopping)
RANDOM_SEED = 42          # 固定随机种子, 保证可复现


def compute_metrics(y_true, proba):
    auc = roc_auc_score(y_true, proba)
    auprc = average_precision_score(y_true, proba)
    return auc, auprc


def cv_eval_one_combo(exp, dataset, cols, lr, depth, subsample):
    """对一组超参做 5 折交叉验证, 返回每折 AUC/AUPRC/best_iteration 以及训练集、验证集的预测"""
    base = data_path + "cross_valid/{}_{}/".format(exp, dataset)
    fold_aucs, fold_auprcs, fold_best_iters = [], [], []
    fold_train_evals, fold_valid_evals = [], []
    for fold in range(N_FOLDS):
        X_train = pd.read_csv(base + "fold{}/train.csv".format(fold))[cols]
        y_train = pd.read_csv(base + "fold{}/train.csv".format(fold))['final_cluster']
        X_valid = pd.read_csv(base + "fold{}/valid.csv".format(fold))[cols]
        y_valid = pd.read_csv(base + "fold{}/valid.csv".format(fold))['final_cluster']

        n_class0, n_class1 = (y_train == 0).sum(), (y_train == 1).sum()
        class_weights = {0: 1.0, 1: n_class0 / n_class1}
        model = CatBoostClassifier(subsample=subsample, learning_rate=lr, max_depth=depth,
                                   eval_metric='AUC', loss_function='Logloss', cat_features=None,
                                   n_estimators=N_ESTIMATORS_CV, class_weights=class_weights, random_seed=RANDOM_SEED, verbose=False)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], use_best_model=True, verbose=False)

        train_proba = model.predict_proba(X_train)[:, 1]
        valid_proba = model.predict_proba(X_valid)[:, 1]
        auc, auprc = compute_metrics(y_valid, valid_proba)
        fold_aucs.append(auc)
        fold_auprcs.append(auprc)
        fold_best_iters.append(model.get_best_iteration())
        fold_train_evals.append(pd.DataFrame({'label': y_train.values, 'pred_prob': train_proba}))
        fold_valid_evals.append(pd.DataFrame({'label': y_valid.values, 'pred_prob': valid_proba}))
    return fold_aucs, fold_auprcs, fold_best_iters, fold_train_evals, fold_valid_evals


def run_cat(exp_train):
    cols = load_prote_sig_cols()
    print("cols: {}".format(len(cols)))

    for clu_list, dataset in [([1, 2], '0_VS_12'), ([1], '0_VS_1'), ([2], '0_VS_2')]:
    # for clu_list, dataset in [([1, 2], '0_VS_12'), ([1], '0_VS_1')]:
    # for clu_list, dataset in [([2], '0_VS_2')]:
        print("=" * 60)
        print("task: {}_{}".format(exp_train, dataset))
        out_dir = result_path + "{}_{}/".format(exp_train, dataset)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # ---------- 1) 网格搜索: 每组超参跑 5 折 CV, 按验证集 AUC 选最优 ----------
        cv_rows, best_key, best_mean_auc = [], None, -1.0
        best_fold_train_evals, best_fold_valid_evals, best_best_iters, best_metrics = None, None, None, None
        for (lr, depth, subsample) in PARAM_GRID:
            fold_aucs, fold_auprcs, fold_best_iters, fold_train_evals, fold_valid_evals = cv_eval_one_combo(exp_train, dataset, cols, lr, depth, subsample)
            mean_auc, std_auc = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
            mean_auprc, std_auprc = float(np.mean(fold_auprcs)), float(np.std(fold_auprcs))
            cv_rows.append({'learning_rate': lr, 'max_depth': depth, 'subsample': subsample,
                            'mean_AUC': mean_auc, 'std_AUC': std_auc,
                            'mean_AUPRC': mean_auprc, 'std_AUPRC': std_auprc})
            print("lr={}, depth={}, sub={} -> AUC {:.4f}±{:.4f}, AUPRC {:.4f}±{:.4f}".format(
                lr, depth, subsample, mean_auc, std_auc, mean_auprc, std_auprc))
            if mean_auc > best_mean_auc:
                best_mean_auc = mean_auc
                best_key = (lr, depth, subsample)
                best_fold_train_evals = fold_train_evals
                best_fold_valid_evals = fold_valid_evals
                best_best_iters = fold_best_iters
                best_metrics = {'mean_AUC': mean_auc, 'std_AUC': std_auc,
                                'mean_AUPRC': mean_auprc, 'std_AUPRC': std_auprc,
                                'fold_AUC': fold_aucs, 'fold_AUPRC': fold_auprcs}

        # 网格搜索结果(5折验证集 AUC/AUPRC 均值±SD)
        pd.DataFrame(cv_rows).to_csv(out_dir + "{}_cv_results.csv".format(dataset), index=None)

        best_lr, best_depth, best_sub = best_key
        best_params = {'learning_rate': best_lr, 'max_depth': best_depth, 'subsample': best_sub,
                       'n_estimators': int(round(np.mean(best_best_iters))) + 1}
        best_params.update(best_metrics)
        with open(out_dir + "{}_best_params.json".format(dataset), "w") as f:
            json.dump(best_params, f, indent=2)
        print("best params: {}".format(best_params))

        # 最优超参下的每折训练集+验证集预测(供 14 画 5折CV图, 全程不含 Test)
        for fold in range(N_FOLDS):
            best_fold_train_evals[fold].to_csv(out_dir + "fold{}_train_eval.csv".format(fold), index=None)
            best_fold_valid_evals[fold].to_csv(out_dir + "fold{}_valid_eval.csv".format(fold), index=None)

        # ---------- 2) 用全部 Discovery 样本训练唯一锁定最终模型 ----------
        base = data_path + "cross_valid/{}_{}/".format(exp_train, dataset)
        df_full = pd.read_csv(base + "full_train.csv")
        X_full = df_full[cols]
        y_full = df_full['final_cluster']

        n_class0, n_class1 = (y_full == 0).sum(), (y_full == 1).sum()
        class_weights = {0: 1.0, 1: n_class0 / n_class1}
        final_model = CatBoostClassifier(subsample=best_sub, learning_rate=best_lr, max_depth=best_depth,
                                         eval_metric='AUC', loss_function='Logloss', cat_features=None,
                                         n_estimators=best_params['n_estimators'], class_weights=class_weights,
                                         random_seed=RANDOM_SEED, verbose=False)
        final_model.fit(X_full, y_full)
        final_model.save_model(out_dir + "{}_{}_final_model.cbm".format(exp_train, dataset))
        print("final model saved, n_estimators={}".format(best_params['n_estimators']))

        full_proba = final_model.predict_proba(X_full)[:, 1]
        full_auc, full_auprc = compute_metrics(y_full, full_proba)
        df_full_eval = pd.DataFrame({'label': y_full.values, 'pred_prob': full_proba,
                                     'pred_label': (full_proba >= 0.5).astype(int)})
        df_full_eval.to_csv(out_dir + "{}_{}_full_eval.csv".format(exp_train, dataset), index=None)
        with open(out_dir + "{}_full_metrics.json".format(dataset), "w") as f:
            json.dump({'AUC': full_auc, 'AUPRC': full_auprc}, f, indent=2)
        print("full AUC={:.4f}, AUPRC={:.4f}".format(full_auc, full_auprc))

        # ---------- 3) Test 直接加载锁定模型预测(不训练/不调参/不CV) ----------
        df_test = pd.read_csv(base + "test.csv")
        X_test = df_test[cols]
        y_test = df_test['final_cluster']
        test_proba = final_model.predict_proba(X_test)[:, 1]
        test_auc, test_auprc = compute_metrics(y_test, test_proba)
        df_test_eval = pd.DataFrame({'label': y_test.values, 'pred_prob': test_proba,
                                     'pred_label': (test_proba >= 0.5).astype(int)})
        df_test_eval.to_csv(out_dir + "{}_{}_test_eval.csv".format(exp_train, dataset), index=None)
        with open(out_dir + "{}_test_metrics.json".format(dataset), "w") as f:
            json.dump({'AUC': test_auc, 'AUPRC': test_auprc}, f, indent=2)
        print("Test AUC={:.4f}, AUPRC={:.4f}".format(test_auc, test_auprc))

        # ---------- 4) SHAP: 用同一个最终模型, 分别计算 Discovery 全集 + Test 集 ----------
        explainer = shap.TreeExplainer(final_model)
        shap_full = explainer.shap_values(X_full)
        shap_test = explainer.shap_values(X_test)
        np.save(out_dir + "{}_full_shap.npy".format(dataset), shap_full)
        np.save(out_dir + "{}_test_shap.npy".format(dataset), shap_test)
        print("SHAP saved")


if __name__ == '__main__':
    # run_cat('Diag_Discovery')
    run_cat('Diag_self_Discovery')
    run_cat('Diag_drug_Discovery')
