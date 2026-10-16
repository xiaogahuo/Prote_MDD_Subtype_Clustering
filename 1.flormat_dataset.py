import pandas as pd
from SUtils import *
from datetime import datetime
from functools import reduce


# ICD10 Disease Columns - -41270   from '41270-0.0' to '41270-0.242'
diagnosis_ICD = ['41270-0.'+str(i) for i in range(243)]
# print(diagnosis_ICD)
# ICD10-Affective-Disorder  Disease Code
code_ICD10_F30 = ['F30', 'F300', 'F301', 'F302', 'F308', 'F309']  # 4
code_ICD10_F31 = ['F31', 'F310', 'F311', 'F312', 'F313', 'F314', 'F315', 'F316', 'F317', 'F318', 'F319']  # 4
code_ICD10_F34 = ['F34', 'F340', 'F341', 'F348', 'F349']  # 4
code_ICD10_F38 = ['F38', 'F380', 'F381', 'F388']  # 4
code_ICD10_F39 = ['F39']  # 4
ICD_Affective_Disorder_set = set(code_ICD10_F30 + code_ICD10_F31 + code_ICD10_F34 + code_ICD10_F38 + code_ICD10_F39)
chronic_code_ICD10 = ['I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I20', 'I21', 'I22', 'I23', 'I24', 'I25', 'E10', 'E11']


def AD_exclude(row):
    diag_lst = row[diagnosis_ICD].tolist()
    diag_lst = [x for x in diag_lst if not pd.isna(x)]
    intersection = ICD_Affective_Disorder_set.intersection(set(diag_lst))
    # print(intersection)
    if len(intersection) == 0:
        return 0  # without AD, keep
    else:
        return 1  # has AD,  will be excluded


codes_mdd = [20, 21, 30, 31, 40, 41, 50, 51]
codes_mdd_self_excluded = [20, 21, 30, 31, 40, 41]
def mdd(v):
    if pd.isna(v):
        return NC  # NC
    elif int(v) in codes_mdd:
        return MDD  # MDD
    else:
        return UNNORMAL  # Un-normal data


def mdd_self_excluded(v):
    if pd.isna(v):
        return NC  # NC
    elif int(v) in codes_mdd_self_excluded:
        return MDD  # MDD
    else:
        return UNNORMAL  # Un-normal data


def base_time(row):
    mdd_tab, diag_time, base_time = row['MDD'], row['130894-0.0'], row['53-0.0']
    if mdd_tab == NC:
        return 0   # NC
    elif pd.isna(diag_time):  # mdd == 1, diag_time == NA, exclude
        return -1  # drop
    elif datetime.strptime(diag_time, "%Y-%m-%d") <= datetime.strptime(base_time, "%Y-%m-%d"):  # 基线之前诊断为抑郁症
        return 1
    else:  # 基线之后诊断为抑郁症
        return 2


# 判断字符串result是否以某个子串tag开头, 统计总数
def startCnt(listTag, listResult):
    num = 0
    for result in listResult:
        for tag in listTag:
            if result.startswith(tag):
                num += 1
    return num


def cnt_chronic_diseases(x):
    # print(x[diagnosis_ICD].tolist())
    result = [str(elem) if not pd.isna(elem) else None for elem in x[diagnosis_ICD].tolist()]
    result = list(filter(None, result))
    chronic_num = startCnt(chronic_code_ICD10, result)
    print("ICD_disease: {}, chronic_num: {}".format(result, chronic_num))
    # return result, chronic_num
    return chronic_num


# 1-3 Drug_Yes    4, 5, -7 Drug_No     others   Drug_Unknown
def drug_intake(x):
    # 6177 male data ; 6153 female data
    result = [x['6177-0.0'], x['6177-0.1'], x['6177-0.2'], x['6153-0.0'], x['6153-0.1'], x['6153-0.2'], x['6153-0.3']]
    result = [x for x in result if x]
    if set([1, 2, 3]).intersection(set(result)):
        return 'Drug_Yes'  # 'Drug_Yes'
    elif set([4, 5, -7]).intersection(set(result)):
        return 'Drug_No'  # 'Drug_No'
    else:
        return 'Drug_Unknown'  # 'Drug_Unknown'


def format_MDD():
    df_prote = pd.read_csv(path + "proteomics.csv", usecols=['eid'])
    print("df_prote: {}".format(df_prote.shape[0]))

    df_PPP = pd.read_csv(root_path + "ukb676487.csv",
                         usecols=['eid', '30903-0.0', '30903-1.0', '30903-2.0', '30903-3.0'])  # excluded
    print("before df_PPP: {}".format(df_PPP.shape[0]))
    df_PPP = df_PPP[(df_PPP['30903-0.0'] != 1) & (df_PPP['30903-1.0'] != 1) & (df_PPP['30903-2.0'] != 1) & (df_PPP['30903-3.0'] != 1)]
    print("after df_PPP: {}".format(df_PPP.shape[0]))
    df_PPP = df_PPP[['eid']]

    df_MDD = pd.read_csv(root_path + "ukb676218.csv", usecols=['eid', '130894-0.0', '130895-0.0', '130896-0.0', '130897-0.0'],
                     dtype={'130894-0.0': str})
    print("df_MDD before merge: {}".format(df_MDD.shape[0]))
    # df_MDD = pd.merge(df_MDD, df_prote, how='inner', on='eid', sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)
    df_MDD = reduce(lambda left, right: pd.merge(left, right, on=['eid'], how='inner'), [df_MDD, df_prote, df_PPP])
    print("df_MDD after merge with df_prote, df_PPP: {}".format(df_MDD.shape[0]))
    values_to_exclude = ['1900-01-01', '1901-01-01', '1902-02-02', '1903-03-03', '1909-09-09', '2037-07-07']
    df_MDD = df_MDD[~df_MDD['130894-0.0'].isin(values_to_exclude)]
    print("df_MDD diag_time excluded with ['1900-01-01', '1901-01-01', '1902-02-02', '1903-03-03', '1909-09-09', '2037-07-07'] : {}".format(df_MDD.shape[0]))
    df_MDD['MDD'] = df_MDD['130895-0.0'].map(mdd)
    df_MDD['MDD_self_excluded'] = df_MDD['130895-0.0'].map(mdd_self_excluded)
    print("df_MDD: {}, NC:{}, MDD:{}, UNNORMAL:{}".format(df_MDD.shape[0], df_MDD[df_MDD['MDD'] == NC].shape[0], df_MDD[df_MDD['MDD'] == MDD].shape[0], df_MDD[df_MDD['MDD'] == UNNORMAL].shape[0]))

    df_chunk = pd.read_csv(root_path+"1.social_demography.csv", chunksize=50000, low_memory=False)
    lst = []
    for i, chunk in enumerate(df_chunk):
        chunk = pd.merge(chunk, df_MDD, how='inner', on='eid', sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)
        lst.append(chunk)
    df = pd.concat(lst, ignore_index=True)
    print("df: {}, NC:{}, MDD:{}".format(df.shape[0], df[df['MDD'] == NC].shape[0], df[df['MDD'] == MDD].shape[0]))
    df['AD_exclude'] = df.apply(lambda row: AD_exclude(row), axis=1)
    df = df[df['AD_exclude'] == 0]
    print("df after exclude Affective-Disorder: {}, NC:{}, MDD:{}".format(df.shape[0], df[df['MDD'] == NC].shape[0], df[df['MDD'] == MDD].shape[0]))
    df["Btime"] = df.apply(lambda row: base_time(row), axis=1)
    print("df Basetime: {}, NC:{}, MDD_before_Btime:{}, MDD_after_Btime:{}, illegal:{}".format(df.shape[0], df[df['Btime'] == 0].shape[0],\
                       df[df['Btime'] == 1].shape[0], df[df['Btime'] == 2].shape[0], df[df['Btime'] == -1].shape[0]))
    chronic_num = df.apply(lambda x: cnt_chronic_diseases(x), axis=1)
    # chronic_diseases, chronic_num = zip(*rst)
    # df['chronic_diseases'] = pd.Series(chronic_diseases)
    df['chronic_num'] = pd.Series(chronic_num)

    df_100045 = pd.read_csv(path + "Cate_100045_filtered.csv")
    print("df_100045: {}".format(df_100045.shape[0]))
    df = reduce(lambda left, right: pd.merge(left, right, on=['eid'], how='inner'), [df, df_100045])
    df['Drug'] = df.apply(lambda x: drug_intake(x), axis=1)
    for col in ['2050-0.0', '2060-0.0', '2070-0.0', '2080-0.0']:
        df[col] = df[col] - 1  # 1, 2, 3, 4 -> 0, 1, 2, 3  item -> Score
        df[col] = df[col].fillna(0)  # fill NAN with 0
        df.loc[df[col] < 0, col] = 0  # fill neg with 0
    df['PHQ'] = df.apply(lambda x: x['2050-0.0'] + x['2060-0.0'], axis=1)
    df['PHQ_Total'] = df.apply(lambda x: x['2050-0.0'] + x['2060-0.0'] + x['2070-0.0'] + x['2080-0.0'], axis=1)
    print("df: {}, NC:{}, MDD:{}".format(df.shape[0], df[df['MDD'] == NC].shape[0], df[df['MDD'] == MDD].shape[0]))
    df.to_csv(path + "Social_MDD.csv", index=None)


if __name__ == '__main__':
    format_MDD()

    # # 检查每个元素是否为空值
    # df = pd.read_csv(path + "Social_MDD.csv", low_memory=False)
    # null_counts = df[['chronic_num']].isnull().sum()
    # # 打印每列的空值数量
    # print(null_counts)
