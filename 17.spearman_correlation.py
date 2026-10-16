from SUtils import *
from scipy.stats import spearmanr

def corr(exp):
    df = pd.read_csv(data_path + f"Linear_Regression_PRS_Dataset_{exp}.csv", low_memory=False)
    print(df.columns.tolist())

    cols = load_prote_sig_cols()
    prs_cols = ['5e08', '1e07', '5e07', '1e06', '5e06', '1e05', '5e05', '1e04', '5e04', '1e03', '5e03', '1e02', '5e02']

    f = open(result_path + f"spearmanr_corr_result_{exp}.csv", "w")
    print("prote_col,PRS_col,all_pval,all_coef,0_pval,0_coef,1_pval,1_coef,2_pval,2_coef,MDD_pval,MDD_coef", file=f)
    for prote_col in cols:
        for PRS_col in prs_cols:
            print(prote_col, PRS_col)
            all_coef, all_pval = spearmanr(df[prote_col], df['SCORE_'+PRS_col])
            coef_0, pval_0 = spearmanr(df[df['final_cluster'].isin([0])][prote_col], df[df['final_cluster'].isin([0])]['SCORE_'+PRS_col])
            coef_1, pval_1 = spearmanr(df[df['final_cluster'].isin([1])][prote_col], df[df['final_cluster'].isin([1])]['SCORE_'+PRS_col])
            coef_2, pval_2 = spearmanr(df[df['final_cluster'].isin([2])][prote_col], df[df['final_cluster'].isin([2])]['SCORE_'+PRS_col])
            coef_MDD, pval_MDD = spearmanr(df[df['final_cluster'].isin([1,2])][prote_col], df[df['final_cluster'].isin([1,2])]['SCORE_'+PRS_col])
            print("{},{},{},{},{},{},{},{},{},{},{},{}".format(prote_col, PRS_col, all_pval, all_coef, pval_0, coef_0, pval_1, coef_1, pval_2, coef_2, pval_MDD, coef_MDD), file=f)


if __name__ == '__main__':
    corr('Diag_Discovery')
    corr('Diag_Test')