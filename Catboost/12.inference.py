from catboost import CatBoostClassifier
from SUtils import *
import shap

def inference(exp, model_name, cols):
    df_MDD = pd.read_csv(data_path + "Diag3_MDD.csv", low_memory=False)
    df_MDD['final_cluster'] = 1

    df_NC = pd.read_csv(data_path + "Diag3_NC.csv", low_memory=False)
    df_NC['final_cluster'] = 0
    df = pd.concat([df_MDD, df_NC])
    print(df.groupby('final_cluster').size())

    model = CatBoostClassifier()
    model.load_model(result_path + "{}/{}.cbm".format(exp, model_name))
    X_infer = df[cols]
    y_infer = df['final_cluster']

    predict_prob = model.predict_proba(X_infer)  # 返回预测概率
    df_infer = pd.DataFrame()
    df_infer['label'] = y_infer
    df_infer['pred_prob'] = predict_prob[:, 1]
    df_infer['pred_label'] = model.predict(X_infer)
    df_infer.to_csv(result_path + "{}/{}_infer_eval.csv".format(exp, model_name), index=None)

    # save SHAP_values
    explainer = shap.TreeExplainer(model)  # #这里的model在准备工作中已经完成建模，模型名称就是model
    shap_values = explainer.shap_values(X_infer)  # 传入特征矩阵X，计算SHAP值
    print("shap_values shape: {}".format(shap_values.shape))
    np.save(result_path + "{}/{}_infer_shap.npy".format(exp, model_name), shap_values)


if __name__ == '__main__':
    model_name = "catboost"
    bonferroni_file = result_path + "t_test_result_bonferroni_Diag7.csv"
    cols = pd.read_csv(bonferroni_file)['col'].tolist()

    inference('0_VS_12', model_name, cols)
    inference('0_VS_1', model_name, cols)
    inference('0_VS_2', model_name, cols)