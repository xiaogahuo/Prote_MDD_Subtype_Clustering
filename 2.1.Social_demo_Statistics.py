# -*- coding: utf-8 -*-
"""
生成 Table 1 风格的社会人口学统计表:
  - 首行: 各组样本量 n (100.00)
  - 分类变量: 变量名占一行(该行的 P-value 列放整体 p 值), 每个类别单独一行 "n (col%)"
  - 连续变量: 单行 "mean (std)", 该行 P-value 列放整体 p 值
  - 分类变量用卡方检验, 连续变量用 Kruskal-Wallis 检验
  - 结果统一放入 DataFrame, 保存为 xlsx
"""
from SUtils import *


def social_stats(exp, group_col, group_order, category_cols, continue_cols, out_path=result_path):
    import os
    os.makedirs(out_path, exist_ok=True)

    df = pd.read_csv(data_path + exp + '.csv')
    print('---------------exp: {}, number: {}------------------------'.format(exp, df.shape[0]))
    table = build_table_statistics(df, group_col, group_order, category_cols, continue_cols)
    out_file = os.path.join(out_path, f'Table1_demographics_{exp}_{group_col}.xlsx')
    table.to_excel(out_file, index=False)
    print(table.to_string(index=False))
    print('已保存: {}'.format(out_file))

if __name__ == '__main__':
    category_cols = cat_cols + cate100057_cols[1:]
    continue_cols = [cate100057_cols[0]] + conti_cols

    # social_stats('Diag_Discovery', 'MDD', [0, 1], category_cols, continue_cols)
    # social_stats('Diag_Test', 'MDD', [0, 1], category_cols, continue_cols)
    # social_stats('Diag_Combine', 'source', ['discovery', 'test'], category_cols, continue_cols)

    social_stats('Diag_self_Discovery', 'MDD', [0, 1], category_cols, continue_cols)
    social_stats('Diag_self_Test', 'MDD', [0, 1], category_cols, continue_cols)
    social_stats('Diag_self_Combine', 'source', ['discovery', 'test'], category_cols, continue_cols)

    social_stats('Diag_drug_Discovery', 'MDD', [0, 1], category_cols, continue_cols)
    social_stats('Diag_drug_Test', 'MDD', [0, 1], category_cols, continue_cols)
    social_stats('Diag_drug_Combine', 'source', ['discovery', 'test'], category_cols, continue_cols)