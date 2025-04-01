import seaborn as sns
import matplotlib.pyplot as plt
from SUtils import *

# 示例数据
df = pd.read_csv(data_path + "Linear_Regression_PRS_Dataset.csv")
df['final_cluster'] = df['final_cluster'].replace({0: 'HC', 1: 'Subtpye1', 2: 'Subtpye2'})
palette = {"HC": "#0984e3", "Subtpye1": "#e17055", "Subtpye2": "#f5cd79"}

cols = ['PRS_SCORE_5e08', 'PRS_SCORE_1e07', 'PRS_SCORE_5e07', 'PRS_SCORE_1e06', 'PRS_SCORE_5e06']
# 遍历子图并绘制小提琴图
for col in cols:
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(6, 6))
    sns.violinplot(hue="final_cluster", hue_order=['HC', 'Subtpye1', 'Subtpye2'], y=col, data=df, palette=palette, legend=False, width=.3, gap=.3)
    ax = plt.gca()
    ax.yaxis.set_major_formatter('{:.3g}'.format)
    plt.savefig(img_path+'violin_{}.png'.format(col), dpi=720)
    plt.show()
    plt.clf()
