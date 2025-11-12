from SUtils import *
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

def feature_select(dataset):
    perform_t_test(dataset)
    sig(dataset)


def perform_t_test(dataset):
    df = pd.read_csv(data_path + "{}.csv".format(dataset))
    df0 = df[df['MDD'] == 0]
    df1 = df[df['MDD'] == 1]
    f = open(result_path + "t_test_result_{}.csv".format(dataset), "w")
    print("col,tstat,pval,M_all,STD_all,M0,STD0,M1,STD1", file=f)
    # for col in prote_cols:
    for col in Prote_cols_all:
        tstat, pval = count_T(col, 'MDD', df)
        M_all = df[col].mean()
        STD_all = df[col].std(ddof=1)
        M0 = df0[col].mean()
        STD0 = df0[col].std(ddof=1)
        M1 = df1[col].mean()
        STD1 = df1[col].std(ddof=1)
        print("{},{},{},{},{},{},{},{},{}".format(col, tstat, pval, M_all, STD_all, M0, STD0, M1, STD1), file=f)


def sig(dataset):
    csv_file = result_path + "t_test_result_{}.csv".format(dataset)
    sig_file = result_path + "t_test_result_sig_{}.csv".format(dataset)
    bernoulli_file = result_path + "t_test_result_bonferroni_{}.csv".format(dataset)
    fdr_file = result_path + "t_test_result_fdr_{}.csv".format(dataset)
    df = pd.read_csv(csv_file)
    df_sig = df[df['pval'] < 0.05]
    df_sig.to_csv(sig_file, index=None)

    df_bernoulli = df[df['pval'] < 0.05/2911]
    df_bernoulli.to_csv(bernoulli_file, index=None)

    bool_r, p_fdr = fdrcorrection(df['pval'].tolist(), alpha=0.05, method='indep', is_sorted=False)
    df['fdr_adj'] = pd.Series(p_fdr)
    df_fdr = df[df['fdr_adj'] < 0.05]
    df_fdr.to_csv(fdr_file, index=None)


# https://hemtools.readthedocs.io/en/latest/content/Bioinformatics_Core_Competencies/Volcanoplot.html
def plot_volcano(dataset):
    df = pd.read_csv(result_path + "t_test_result_{}.csv".format(dataset))

    plt.scatter(x=df['M1'], y=df['pval'].apply(lambda x: -np.log10(x)), s=1, label="Not significant")

    # highlight up regulated prote
    up = df[(df['pval'] < 0.05/2911)]

    plt.scatter(x=up['M1'], y=up['pval'].apply(lambda x: -np.log10(x)), s=2, label="Bonferroni adjusted significant", color="red")
    plt.xlabel("Mean of Z_score for MDD")
    plt.ylabel("-logPvalue")
    plt.axhline(-np.log10(0.05/2911), color="grey", linestyle="--")
    plt.legend()
    plt.savefig(img_path + "volcano_prote2911_{}.png".format(dataset))
    plt.clf()

if __name__ == '__main__':
    # feature_select("DiagA")
    # feature_select("DiagB")
    # feature_select("Diag")
    plot_volcano("DiagA")
    plot_volcano("DiagB")
    plot_volcano("Diag")

    # cols_sigA = pd.read_csv(result_path + "t_test_result_bonferroni_DiagA.csv")
    # cols_sigB = pd.read_csv(result_path + "t_test_result_bonferroni_DiagB.csv")
    # cols_sig = pd.merge(cols_sigA, cols_sigB, how='inner', on='col', suffixes=('_A', '_B'), copy=True, indicator=False)
    # cols_sig.to_csv(result_path + "t_test_result_bonferroni_DiagA&B.csv", index=None)



