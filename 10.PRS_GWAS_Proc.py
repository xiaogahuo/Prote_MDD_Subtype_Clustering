import pandas as pd
from SUtils import *
import sys


def save_dataframe_in_chunks(df, chr, s_name, chunksize=9990):
    """
    将pandas DataFrame按指定的行数分割并保存为多个文件。

    参数：
    df (pd.DataFrame): 需要分割的DataFrame
    output_prefix (str): 输出文件的前缀，每个文件的名称将是前缀加上序号
    chunksize (int): 每个文件包含的行数，默认为9000
    """
    start = 0
    end = chunksize
    file_count = 1

    while start < df.shape[0]:
        # 获取当前块的数据
        chunk = df[start:end]
        # 保存当前块到文件
        chunk.to_csv(result_path + "GWAS/{}/GWAS_Betas_{}_{}_{}.csv".format(s_name, s_name, chr, file_count), index=None)

        # 更新起始和结束索引以及文件计数器
        start += chunksize
        end += chunksize
        file_count += 1
    print("数据保存完成: {}, 染色体chr: {}, 文件个数{}".format(s_name, chr, file_count-1))

def filter_sig_MDD(threshold, s_name):
    df_exp = pd.read_csv(path + mdd_gwas_summary_file, sep=' ', dtype={'CHR': str, 'BP': str})
    # print(df_exp.columns.tolist())
    print("before P < {} filter: {}".format(s_name, df_exp.shape[0]))
    df_exp = df_exp[df_exp['P'] < threshold]
    print("after  P < {} filter: {}".format(s_name, df_exp.shape[0]))
    df_exp['CHR:BP'] = df_exp[['CHR', 'BP']].apply(lambda x: ':'.join([str(v) for v in x]), axis=1)
    df_exp['A1_FRQ'] = df_exp[['FRQ_A_166773', 'FRQ_U_507679']].apply(lambda x:  (x['FRQ_A_166773']*166773+x['FRQ_U_507679']*507679)/(166773+507679), axis=1)
    df_exp['BETA'] = df_exp['OR'].apply(lambda x: np.log(x))
    df_exp = df_exp[['SNP', 'CHR', 'BP', 'A1', 'A2', 'BETA', 'A1_FRQ', 'CHR:BP', 'P']]
    df_exp = df_exp.rename(columns={'SNP': 'rsid', 'CHR': 'chr_name', 'BP': 'chr_position', 'A1': 'effect_allele', 'A2': 'noneffect_allele', 'BETA': 'Beta', 'A1_FRQ': 'eaf', 'CHR:BP': 'chr_pos'})
    if not os.path.exists(result_path + "GWAS/{}/".format(s_name)):
        os.makedirs(result_path + "GWAS/{}/".format(s_name))
    df_exp.to_csv(result_path + "GWAS/{}/GWAS_Betas_{}.csv".format(s_name, s_name), index=None)

    # 使用groupby函数对DataFrame进行分组
    groups = df_exp.groupby('chr_name')
    for chr, group in groups:
        print("{} : {}".format(chr, group.shape[0]))
        save_dataframe_in_chunks(group, chr, s_name)


# ukbconv ukb669272.enc_ukb csv -ifields_id_Sample_QC.txt -oSample_QC    获取 Sample_QC.csv
def Sample_QC_IDS():
    df = pd.read_csv(root_path + "Sample_QC.csv")
    print(df.shape[0])
    df = df[(df["22006-0.0"] == 1) & (df["22020-0.0"] == 1)]   # 22006-0.0 代表高加索人群
    print(df.shape[0])
    df.to_csv(result_path + "usedinpca.txt", index=None, columns=['eid'])


#       #FID      IID  ALLELE_CT  NAMED_ALLELE_DOSAGE_SUM  SCORE1_AVG
# 0  2046642  2046642       1490                  768.988    0.000274
# 1  4270668  4270668       1490                  905.463   -0.002423
# 2  5053312  5053312       1490                  704.180   -0.000345
# 3  1780405  1780405       1490                  721.000   -0.001148
# 4  3030933  3030933       1490                  734.004   -0.000045
def PRS_score_kwtest(PRS_file, exp, clust_N):
    df_PRS = pd.read_csv(PRS_path + "{}_PRS.sscore".format(PRS_file), sep=r'[ \t]', engine='python')
    df_PRS.rename(columns={'#FID': 'eid'}, inplace=True)
    # print(df_PRS.head())
    print("************************************************"+"{}, {}: {}".format(exp, PRS_file, clust_N)+"************************************************")
    df_MDD, df_NC = pd.read_csv(data_path + f"{exp}_MDD.csv", low_memory=False), pd.read_csv(data_path + f"{exp}_NC.csv", low_memory=False)
    df_NC['final_cluster'] = 0
    df_Y = pd.read_csv(result_path + "{}/{}/{}_components_cluster_rst_{}.csv".format(exp, "NMF", 2, 2), low_memory=False, usecols=['eid', 'final_cluster'])

    if (clust_N == '0vs1'):
        df_Y = df_Y[df_Y['final_cluster'] == 1]
        df_MDD = pd.merge(df_MDD, df_Y, on='eid', how='inner')
        df = pd.concat([df_MDD, df_NC])
    elif (clust_N == '0vs2'):
        df_Y = df_Y[df_Y['final_cluster'] == 2]
        df_MDD = pd.merge(df_MDD, df_Y, on='eid', how='inner')
        df = pd.concat([df_MDD, df_NC])
    else:  # clust_N == '1vs2'
        df = pd.merge(df_MDD, df_Y, on='eid', how='inner')
    df = pd.merge(df, df_PRS, on='eid', how='inner')
    print(df.groupby('final_cluster').size())
    kwtest('SCORE1_AVG', 'final_cluster', df)


if __name__ == '__main__':
    # filter_sig_MDD(5e-08, '5e08')
    # filter_sig_MDD(1e-07, '1e07')
    # filter_sig_MDD(5e-07, '5e07')
    # filter_sig_MDD(1e-06, '1e06')
    # filter_sig_MDD(5e-06, '5e06')
    # filter_sig_MDD(1e-05, '1e05')
    # filter_sig_MDD(5e-05, '5e05')
    # filter_sig_MDD(1e-04, '1e04')
    # filter_sig_MDD(5e-04, '5e04')
    # filter_sig_MDD(1e-03, '1e03')
    # filter_sig_MDD(5e-03, '5e03')
    # filter_sig_MDD(1e-02, '1e02')
    # filter_sig_MDD(5e-02, '5e02')
    # Sample_QC_IDS()

    for f in ['5e08', '1e07', '5e07', '1e06', '5e06', '1e05','5e05','1e04','5e04','1e03','5e03','1e02','5e02']:
        PRS_score_kwtest(f, 'Diag', '0vs1')
        PRS_score_kwtest(f, 'Diag', '0vs2')
        PRS_score_kwtest(f, 'Diag', '1vs2')
        PRS_score_kwtest(f, 'DiagA', '0vs1')
        PRS_score_kwtest(f, 'DiagA', '0vs2')
        PRS_score_kwtest(f, 'DiagA', '1vs2')
        PRS_score_kwtest(f, 'DiagB', '0vs1')
        PRS_score_kwtest(f, 'DiagB', '0vs2')
        PRS_score_kwtest(f, 'DiagB', '1vs2')