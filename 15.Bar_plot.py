import seaborn as sns
import matplotlib.pyplot as plt
from SUtils import *

# 示例数据
df = pd.read_csv(data_path + "Linear_Regression_PRS_Dataset.csv")
df['final_cluster'] = df['final_cluster'].replace({0: 'HC', 1: 'Subtpye1', 2: 'Subtpye2'})
palette = {"HC": "#0984e3", "Subtpye1": "#e17055", "Subtpye2": "#fab1a0"}

cols = ['PRS_SCORE_5e08', 'PRS_SCORE_1e07', 'PRS_SCORE_5e07', 'PRS_SCORE_1e06', 'PRS_SCORE_5e06']
# 创建一个5x1的子图布局
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
# 调整子图之间的间距
# fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.subplots_adjust(left=0.2, right=0.85, bottom=0, top=1.0, wspace=0.1, hspace=0)

# 遍历子图并绘制小提琴图
for i, (col, ax) in enumerate(zip(cols, axes.flatten())):
    sns.barplot(hue="final_cluster", hue_order=['HC', 'Subtpye1', 'Subtpye2'], y=col, data=df,
                palette=palette, ax=ax, legend=False, width=.6, gap=.3, capsize=.2,
                err_kws={"color": ".5", "linewidth": 2.5})
    ax.set_title(col)
    # 隐藏X轴列名
    ax.set(xlabel='')
    ax.set(ylabel='')
# 调整子图间距
# plt.tight_layout()

# 显示图形
plt.savefig(img_path+'violin_PRS_score.png', dpi=720)
plt.show()
