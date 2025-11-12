import seaborn as sns
import matplotlib.pyplot as plt
from SUtils import *

# 示例数据
def plot(exp):
    df = pd.read_csv(data_path + f"Linear_Regression_PRS_Dataset_{exp}.csv")
    df['final_cluster'] = df['final_cluster'].replace({0: 'HC', 1: 'Subtpye1', 2: 'Subtpye2'})
    palette = {"HC": "#0984e3", "Subtpye1": "#e17055", "Subtpye2": "#f5cd79"}

    cols = ['SCORE_5e08', 'SCORE_1e07', 'SCORE_5e07', 'SCORE_1e06', 'SCORE_5e06']
    # 遍历子图并绘制小提琴图
    for col in cols:
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.figure(figsize=(6, 6))
        sns.violinplot(hue="final_cluster", hue_order=['HC', 'Subtpye1', 'Subtpye2'], y=col, data=df, palette=palette, legend=False, width=.3, gap=.3)
        ax = plt.gca()
        ax.yaxis.set_major_formatter('{:.3g}'.format)
        plt.savefig(img_path+f'violin_{col}_{exp}.png', dpi=720)
        plt.show()
        plt.clf()

if __name__ == '__main__':
    plot('Diag')
    plot('DiagA')
    plot('DiagB')