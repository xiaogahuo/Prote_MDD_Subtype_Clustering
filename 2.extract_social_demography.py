import pandas as pd
from sklearn.model_selection import train_test_split
from SUtils import *

def social_stats(exps):
    for exp in exps:
        df = pd.read_csv(data_path + exp + '.csv')
        print('---------------exp: {}, number: {}------------------------'.format(exp, df.shape[0]))
        for col in cat_cols+['MDD']:
            count_1feature(col, df)
        for col in conti_cols:
            count_mean_std(col, df)

        for col in cat_cols:
            count_2features(col, 'MDD', df)
            count_Chi2(col, 'MDD', df)
        for col in conti_cols:
            count_mean_std_group_by_div_feature(col, 'MDD', df)
            kwtest(col, 'MDD', df)

##############################################
#
# 执行数据预处理，包括计算随访时长、家族史
# "189-0.0": "Townsend_index"  Townsend deprivation index at recruitment Q1 (≤ -3.63) ;Q2 (-3.63 - -2.12);Q3 (-2.12–0.58) ;Q4 (>0.58);Unknown
# "1558-0.0": "Alcohol_freq",
# "6138-0.0": "Qualifications",    Qualifications  Non-university ;University ;Unknown
#    1	College or University degree
#    2	A levels/AS levels or equivalent
#    3	O levels/GCSEs or equivalent
#    4	CSEs or equivalent
#    5	NVQ or HND or HNC or equivalent
#    6	Other professional qualifications eg: nursing, teaching
#    -7	None of the above
#    -3	Prefer not to answer
# "20116-0.0": "Smoking_status",
# "20117-0.0": "Alcohol drinker status",
# "21000-0.0": "Ethnic_background",   Non-White; White; Unknown   # https://biobank.ndph.ox.ac.uk/showcase/coding.cgi?id=1001
# "21001-0.0": "BMI",  <18.5 kg/m2 ;18.5–24.9 kg/m2;25.0–29.9 kg/m2 ;≥30 kg/m2 ;Unknown
# "21003-0.0": "Age",  <45  45-59  >=60
# "22032-0.0": "IPAQ"  Low Moderate High  Unknown
#
##############################################
# 0 Low 1 Moderate 2 High  Unknown
def sex(x):
    val = x['31-0.0']
    if pd.isnull(val):
        return 'L3_Unknown'
    elif val == 0:
        return 'L1_Female'
    elif val == 1:
        return 'L2_Male'

# 0 Low 1 Moderate 2 High  Unknown
def ipaq(x):
    val = x['22032-0.0']
    if pd.isnull(val):
        return 'L4_Unknown'
    elif val == 0:
        return 'L1_Low'
    elif val == 1:
        return 'L2_Moderate'
    else:
        return 'L3_High'


#  <45  45-59  >=60
def age(x):
    val = x['21003-0.0']
    if val < 45:
        return 'Age1'
    elif 45 <= val < 60:
        return 'Age2'
    else:
        return 'Age3'


# <60 >=60
def age2(x):
    val = x['21003-0.0']
    if val < 60:
        return 'Age1'
    else:
        return 'Age2'


# <18.5 kg/m2 ;18.5–24.9 kg/m2;25.0–29.9 kg/m2 ;≥30 kg/m2 ;Unknown
def bmi(x):
    val = x['21001-0.0']
    if pd.isnull(val):
        return 'BMI5_Unknown'
    elif val < 18.5:
        return 'BMI2'
    elif 18.5 <= val < 25.0:
        return 'BMI1'
    elif 25.0 <= val < 30:
        return 'BMI3'
    else:
        return 'BMI4'

def ethnic_background(x): # Non-White; White; Unknown
    val = x['21000-0.0']
    if pd.isnull(val) or val in [-1, -3]:
        return 'L3_Unknown'
    elif val in [1, 1001, 2001, 3001, 4001]:
        return 'L2_White'
    else:
        return 'L1_Non-White'


def qualifications(x): # Non-university ;University ;Unknown
    val = x['6138-0.0']
    if pd.isnull(val) or val in [-7, -3]:
        return 'L3_Unknown'
    elif val == 1:
        return 'L2_University'
    else:
        return 'L1_Non-university'


def smoking(x): # -3	Prefer not to answer ;  0 Never ; 1	Previous; 2	Current;
    val = x['20116-0.0']
    if pd.isnull(val) or val == -3:
        return 'L4_Unknown'
    elif val == 0:
        return 'L1_Never'
    elif val == 1:
        return 'L2_Previous'
    else:
        return 'L3_Current'


def alcohol(x): # -3	Prefer not to answer ;  0 Never ; 1	Previous; 2	Current;
    val = x['20117-0.0']
    if pd.isnull(val) or val == -3:
        return 'L4_Unknown'
    elif val == 0:
        return 'L1_Never'
    elif val == 1:
        return 'L2_Previous'
    else:
        return 'L3_Current'


def townsend_index(x):
    val = x['189-0.0']
    if pd.isnull(val):
        return 'Q5_unknown'
    elif val <= -3.63:
        return 'Q1'
    elif -3.63 < val <= -2.12:
        return 'Q2'
    elif -2.12 < val <= 0.58:
        return 'Q3'
    else:
        return 'Q4'

field_20107 = ['20107-0.0', '20107-0.1', '20107-0.2', '20107-0.3', '20107-0.4', '20107-0.5', '20107-0.6', '20107-0.7', '20107-0.8', '20107-0.9', '20107-1.0', '20107-1.1', '20107-1.2', '20107-1.3', '20107-1.4', '20107-1.5', '20107-1.6', '20107-1.7', '20107-1.8', '20107-1.9', '20107-2.0', '20107-2.1', '20107-2.2', '20107-2.3', '20107-2.4', '20107-2.5', '20107-2.6', '20107-2.7', '20107-2.8', '20107-2.9', '20107-3.0', '20107-3.1', '20107-3.2', '20107-3.3', '20107-3.4', '20107-3.5', '20107-3.6', '20107-3.7', '20107-3.8', '20107-3.9']
field_20110 = ['20110-0.0', '20110-0.1', '20110-0.2', '20110-0.3', '20110-0.4', '20110-0.5', '20110-0.6', '20110-0.7', '20110-0.8', '20110-0.9', '20110-0.10', '20110-1.0', '20110-1.1', '20110-1.2', '20110-1.3', '20110-1.4', '20110-1.5', '20110-1.6', '20110-1.7', '20110-1.8', '20110-1.9', '20110-1.10', '20110-2.0', '20110-2.1', '20110-2.2', '20110-2.3', '20110-2.4', '20110-2.5', '20110-2.6', '20110-2.7', '20110-2.8', '20110-2.9', '20110-2.10', '20110-3.0', '20110-3.1', '20110-3.2', '20110-3.3', '20110-3.4', '20110-3.5', '20110-3.6', '20110-3.7', '20110-3.8', '20110-3.9', '20110-3.10']
field_20111 = ['20111-0.0', '20111-0.1', '20111-0.2', '20111-0.3', '20111-0.4', '20111-0.5', '20111-0.6', '20111-0.7', '20111-0.8', '20111-0.9', '20111-0.10', '20111-0.11', '20111-1.0', '20111-1.1', '20111-1.2', '20111-1.3', '20111-1.4', '20111-1.5', '20111-1.6', '20111-1.7', '20111-1.8', '20111-1.9', '20111-1.10', '20111-1.11', '20111-2.0', '20111-2.1', '20111-2.2', '20111-2.3', '20111-2.4', '20111-2.5', '20111-2.6', '20111-2.7', '20111-2.8', '20111-2.9', '20111-2.10', '20111-2.11', '20111-3.0', '20111-3.1', '20111-3.2', '20111-3.3', '20111-3.4', '20111-3.5', '20111-3.6', '20111-3.7', '20111-3.8', '20111-3.9', '20111-3.10', '20111-3.11']
family_history_column = field_20107 + field_20110 + field_20111
# coverate_family_history = ['20107-', '20110-', '20111-']  # if contain 12  set 1   others   set 0
def family_history(x):
    if 12 in x[family_history_column].tolist():
        return 'YES' # 有家族史
    else:
        return 'NO'  # 无家族史

# 算随访时间
# 死亡['40000-0.0', '40000-1.0']  失访 ['191-0.0']  Affective-Disorder 分类4  取最早时间 算随访时长
# Affective-Disorder 分类1 2 3 拿对应的诊断时间算随访时长
# 其他受试拿当前时间算随访时长
# 入组时间 ['53-0.0']
def follow_Duration(row):
    cur_date = '2023-11-23'
    if row['MDD'] == NC:
        temp = [str(t) for t in row[['40000-0.0', '40000-1.0', '191-0.0']].tolist()]
        temp.append(cur_date)
        target_date = min([t for t in temp if not pd.isnull(t)])
    else:  # MDD
        target_date = str(row['130894-0.0'])
    target_date = datetime.strptime(target_date, "%Y-%m-%d")
    base_date = datetime.strptime(str(row['53-0.0']), "%Y-%m-%d")
    return (target_date-base_date).days/30


def get_preprocess_data():
    df = pd.read_csv(path + "Social_MDD.csv", low_memory=False)
    df['follow_Duration'] = df.apply(lambda x: follow_Duration(x), axis=1)
    df['Townsend_index_cate'] = df.apply(lambda x: townsend_index(x), axis=1)
    df['Qualifications_cate'] = df.apply(lambda x: qualifications(x), axis=1)
    df['Smoking_status_cate'] = df.apply(lambda x: smoking(x), axis=1)
    df['Alcohol_status_cate'] = df.apply(lambda x: alcohol(x), axis=1)
    # df['Ethnic_background_cate'] = df.apply(lambda x: ethnic_background(x), axis=1)
    df['BMI_cate'] = df.apply(lambda x: bmi(x), axis=1)
    # df['Age_cate'] = df.apply(lambda x: age(x), axis=1)
    # df['Age_cate2'] = df.apply(lambda x: age2(x), axis=1)
    df = df.rename(columns={'21003-0.0': 'Age'})
    df['IPAQ_cate'] = df.apply(lambda x: ipaq(x), axis=1)
    df['family_history'] = df.apply(lambda x: family_history(x), axis=1)
    df['Sex_cate'] = df.apply(lambda x: sex(x), axis=1)
    df.to_csv(path+"Social_MDD_format.csv", index=None)
    df = df[['eid', 'MDD', 'Btime', 'PHQ', 'PHQ_Total']+covr_cols+cate100042_cols+cate100057_cols+cate100060_cols+cate100061_cols+cate100065_cols+cate17518_cols+cate145_cols]
    df_prote = pd.read_csv(path + "proteomics.csv")
    # 计算每列的空值比例
    null_ratios = df_prote.isnull().sum() / df_prote.shape[0]
    # 筛选出空值比例小于等于20%的列
    columns_to_keep = null_ratios[null_ratios <= 0.2].index
    print(columns_to_keep.tolist())
    print(len(columns_to_keep.tolist()))
    # 保留空值比例小于等于20%的列
    df_prote = df_prote[columns_to_keep]  # 2911 columns
    # 选择数值列, 对每个数值列进行 Z 变换
    cols_p = df_prote.columns.tolist()[2:]  # excluded: 'Unnamed: 0', 'eid'
    for col in cols_p:
        df_prote[col] = (df_prote[col] - df_prote[col].mean())/df_prote[col].std()
    df_prote.fillna(0, inplace=True)  # 空值补零- 即均值

    df = pd.merge(df, df_prote, how='inner', on='eid', sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)
    df.to_csv(data_path + "ALL.csv", index=None)

    # df_cox = df[(df['Btime'] == 2) | (df['Btime'] == 0)]
    # df_cox.to_csv(data_path + "Cox.csv", index=None)
    # print("df_cox: {}, NC:{}, MDD:{}".format(df_cox.shape[0], df_cox[df_cox['MDD'] == NC].shape[0], df_cox[df_cox['MDD'] == MDD].shape[0]))
    #
    # df_forcast = df[(df['Btime'] == 2) | ((df['Btime'] == 0) & (df['PHQ'] < 2))]
    # df_forcast.to_csv(data_path + "Forcast.csv", index=None)
    # print("df_forcast: {}, NC:{}, MDD:{}".format(df_forcast.shape[0], df_forcast[df_forcast['MDD'] == NC].shape[0],
    #                                          df_forcast[df_forcast['MDD'] == MDD].shape[0]))

    df_diag = df[((df['Btime'] == 1) & (df['PHQ'] >= 2)) | ((df['Btime'] == 0) & (df['PHQ'] < 2))]
    df_diag.to_csv(data_path + "Diag.csv", index=None)
    print("df_diag: {}, NC:{}, MDD:{}".format(df_diag.shape[0], df_diag[df_diag['MDD'] == NC].shape[0],
                                             df_diag[df_diag['MDD'] == MDD].shape[0]))


import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split_even(df, group_col, test_size=0.5, random_state=0):
    """
    把 df 按 group_col 分层拆成两份，允许 singleton 类存在。
    返回 df_A, df_B
    """
    # 1. 计算每类样本数
    vc = df[group_col].value_counts()
    singleton_cls = vc[vc == 1].index        # 独苗类
    normal_cls    = vc[vc >= 2].index        # 正常类

    # 2. 独苗类单独处理：交替分配到两份，保证数量差 ≤1
    singleton_df = df[df[group_col].isin(singleton_cls)]
    other_df     = df[df[group_col].isin(normal_cls)]

    # 随机打乱顺序
    singleton_df = singleton_df.sample(frac=1, random_state=random_state)
    # 交替分配
    df_A_extra = singleton_df.iloc[::2]
    df_B_extra = singleton_df.iloc[1::2]

    # 3. 正常类用 stratify 拆分
    if len(other_df) > 0:
        df_A_normal, df_B_normal = train_test_split(
            other_df,
            test_size=0.5,
            stratify=other_df[group_col],
            random_state=random_state
        )
    else:
        df_A_normal = df_B_normal = other_df

    # 4. 合并
    df_A = pd.concat([df_A_normal, df_A_extra], ignore_index=True)
    df_B = pd.concat([df_B_normal, df_B_extra], ignore_index=True)

    return df_A, df_B


if __name__ == '__main__':
    # # get_preprocess_data()
    # #
    # # df = pd.read_csv(data_path + "{}.csv".format("Diag"))
    # # df_MDD, df_NC = df[df['MDD'] == MDD], df[df['MDD'] == NC]
    # # df_MDD.to_csv(data_path + "{}_MDD.csv".format("Diag"), index=None)
    # # df_NC.to_csv(data_path + "{}_NC.csv".format("Diag"), index=None)
    # df_MDD, df_NC = pd.read_csv(data_path + "Diag_MDD.csv", low_memory=False), pd.read_csv(data_path + "Diag_NC.csv", low_memory=False)
    # # print(df_MDD.columns.tolist())
    # # df_MDD['group_ID'] = df_MDD['Sex_cate'].astype(str) + '_' + df_MDD['Age'].astype(str)
    # # df_NC['group_ID'] = df_NC['Sex_cate'].astype(str) + '_' + df_NC['Age'].astype(str)
    # # # ---------------- 用法示例 ----------------
    # # df_MDD_A, df_MDD_B = stratified_split_even(df_MDD, group_col='group_ID', test_size=0.5, random_state=0)
    # # df_NC_A, df_NC_B = stratified_split_even(df_NC, group_col='group_ID', test_size=0.5, random_state=0)
    #
    # # print("MDD:{}, NC:{}".format(df_MDD.shape[0], df_NC.shape[0]))
    # df_MDD_A, df_MDD_B = (lambda x: x.sample(frac=1, random_state=0).pipe(lambda y: (y.iloc[:int(len(y) * 0.7)], y.iloc[int(len(y) * 0.7):])))(df_MDD)
    # # print("MDD_A:{}, MDD_B:{}".format(df_MDD_A.shape[0], df_MDD_B.shape[0]))
    # df_NC_A, df_NC_B = (lambda x: x.sample(frac=1, random_state=0).pipe(lambda y: (y.iloc[:int(len(y) * 0.7)], y.iloc[int(len(y) * 0.7):])))(df_NC)
    # # print("NC_A:{}, NC_B:{}".format(df_NC_A.shape[0], df_NC_B.shape[0]))
    # df_MDD_A.to_csv(data_path + "DiagA_MDD.csv", index=None)
    # df_MDD_B.to_csv(data_path + "DiagB_MDD.csv", index=None)
    # df_NC_A.to_csv(data_path + "DiagA_NC.csv", index=None)
    # df_NC_B.to_csv(data_path + "DiagB_NC.csv", index=None)
    # df_A = pd.concat([df_MDD_A, df_NC_A])
    # df_B = pd.concat([df_MDD_B, df_NC_B])
    # df_A.to_csv(data_path + "DiagA.csv", index=None)
    # df_B.to_csv(data_path + "DiagB.csv", index=None)
    # print("Diag_A:{}, Diag_B:{}".format(df_A.shape[0], df_B.shape[0]))

    # 4. 社会人口学数据统计
    exps = ['Diag', 'DiagA', 'DiagB']
    social_stats(exps)




