import pandas as pd

from SUtils import *

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    # 绘制森林图
    fig, ax = plt.subplots(figsize=(10, 6))

    df = pd.read_csv(result_path+'LR_PRS_rst_OR_CI95.csv')
    print(df.head(10))
    # 绘制点和置信区间
    sns.scatterplot(x='OR', y='y_val', hue='Clust', data=df, ax=ax, palette=["green", "orange"])
    # 遍历数据，绘制水平线
    for left, right, y, clust in zip(df['CI_lower'], df['CI_upper'], df['y_val'], df['Clust']):
        if clust == 'clust1':
            ax.hlines(y=y, xmin=left, xmax=right, color='green')
        else:
            ax.hlines(y=y, xmin=left, xmax=right, color='orange')

    # 添加参考线
    ax.axvline(x=1, color='gray', linestyle='--')

    # 不显示图例
    plt.legend().set_visible(False)

    # 设置标签和标题
    ax.set_ylabel('SNP threshold p-value')
    ax.set_xlabel('95% CI')
    ax.set_title('PRS Linear Regression')
    plt.savefig(img_path+'forest.png')



    # # # 显示图形
    # # plt.show()
    # penguins = sns.load_dataset("penguins")
    # print(penguins)
    # penguins.to_csv(result_path + 'penguins.csv')
    # sns.pointplot(data=penguins, x="island", y="body_mass_g", hue="sex")

