import pandas as pd
from SUtils import *

exp_name = "GWAS_Prote_MDD_Subtype_Clustering"

# .fam文件是储存样本信息的文件，为没有标题行的文本文件，每个样本占一行，共有以下6列：
# 1. 家系ID, Family ID (FID)
# 2. 个体ID, Within-family ID (IID;不能为0)
# 3. 父本个体ID(如果父本不在样本中则为0)
# 4. 母本个体ID(如果母本不在样本中则为0)
# 5. 性别(1为雄性,2为雌性, 0表示未知)
# 6. 表型值 (1为对照组,2为病例组,-9、0和非数值在二分类性状时表示缺失值)。如果存在除-9，0，1，2以外的任何数字表型值，则表型为数量性状，而不是病例/对照的二分类性状。在这种情况下，-9仍表示表型缺失值。
def split_cluster(l0, l1, f_name):
    df_MDS = pd.read_csv(root_path + "Sample_QC.csv")
    print("Sample_MDS_QC: {}".format(df_MDS.shape[0]))
    df_MDS = df_MDS[df_MDS["22006-0.0"] == 1]  # 22006-0.0 代表高加索人群   22020-0.0 Used in genetic principal components
    print("Sample_MDS_QC after filter: {}".format(df_MDS.shape[0]))

    rst_file = result_path + "{}/{}_components_cluster_rst_{}.csv".format("NMF", 2, 2)
    df_Y = pd.read_csv(rst_file, low_memory=False, usecols=['eid', 'final_cluster'])
    print(df_Y.groupby('final_cluster').size())

    df_NC = pd.read_csv(data_path + "Diag_NC.csv", low_memory=False, usecols=['eid'])
    df_NC['final_cluster'] = 0
    df = pd.concat([df_NC, df_Y])

    df_bed_sample = pd.read_csv(bed_path + "raw_data/c1.fam", sep=r'[ \t]', header=None, engine='python')
    df_bed_sample.rename(columns={0: 'fid', 1: 'eid', 2: 'pid', 3: 'mid', 4: 'sid'}, inplace=True)
    print(df_bed_sample.head())
    print(df_bed_sample.shape[0])

    df = reduce(lambda left, right: pd.merge(left, right, on=['eid'], how='inner'), [df, df_bed_sample, df_MDS])
    print("df after merge: {}".format(df.shape[0]))

    # 0 vs 1  ----- 0为对照组，1为病例组
    df_0, df_1 = df[df['final_cluster'].isin(l0)], df[df['final_cluster'].isin(l1)]
    df_0['final_cluster'], df_1['final_cluster'] = 0, 1
    df = pd.concat([df_0, df_1])
    print(df.groupby('final_cluster').size())

    df.rename(columns={'fid': '#FID', 'eid': 'IID'}, inplace=True)
    df = df.sort_values(by='IID')  # 匹配sample样本的顺序,方便后续进行GWAS分析
    if not os.path.exists(bed_path + "{}/{}/".format(exp_name, f_name)):
        os.makedirs(bed_path + "{}/{}/".format(exp_name, f_name))
    df.to_csv(bed_path + "{}/{}/{}.csv".format(exp_name, f_name, f_name), columns=['#FID', 'IID'], sep='\t',
              header=None, index=False)
    df.to_csv(bed_path + "{}/{}/{}_pheno.txt".format(exp_name, f_name, f_name),
              columns=['#FID', 'IID', 'pid', 'mid', 'sid', 'final_cluster'], sep='\t', header=True, index=False)


def cnt_sig_pvalue(f_s):
    for file_name in f_s:
        df = []
        for chr in range(1, 23):
            df_chr = pd.read_csv(bed_path + "{}/{}/Step2_SPAtests_c{}.txt".format(exp_name, file_name, chr), sep=r'\s+',
                                 engine='python')
            df.append(df_chr)
        df = pd.concat(df)
        print(df.head())
        df.to_csv(bed_path + "{}/{}/SAIGE_results.txt".format(exp_name, file_name), sep='\t', index=False)
        df = df.sort_values(by="p.value")
        df.to_csv(bed_path + "{}/{}/SAIGE_results.sig.csv".format(exp_name, file_name), sep=',', index=False)  # sep='\t'


if __name__ == '__main__':
    # # before GWAS
    # split_cluster([0], [1, 2], '0_VS_12')
    # split_cluster([0], [1], '0_VS_1')
    # split_cluster([0], [2], '0_VS_2')
    # split_cluster([2], [1], '2_VS_1')

    #after GWAS
    f_s = ["0_VS_12", "0_VS_1", "0_VS_2", "2_VS_1"]
    cnt_sig_pvalue(f_s)

