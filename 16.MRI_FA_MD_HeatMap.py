from SUtils import *


def plot_heatmap(t):
    df = pd.read_excel(result_path + "{}_Heatmap.xlsx".format(t), sheet_name=0, index_col=0)  # Estimate
    df_annot = pd.read_excel(result_path + "{}_Heatmap.xlsx".format(t), sheet_name=1, index_col=0)  # Pvalue
    # 转化为annot 用于后续谱系图的绘制
    # * P<0.05;
    # ** P<0.01;
    # *** P<0.001
    for c in ['Subtpye1_L','Subtpye1_R','Subtype2_L','Subtype2_R']:
        df_annot[c] = df_annot[c].apply(lambda x: annot_trans(x))
    plt.figure(figsize=(6, 18))
    myPlot = sns.heatmap(df, annot=df_annot, fmt='s', annot_kws={"size": 16}, yticklabels=False, cmap="vlag", center=0, vmin=-0.6, vmax=0.6, linewidth=.5)
    myPlot.xaxis.tick_top()
    plt.savefig(img_path + "{}_Heatmap.png".format(t))
    plt.clf()


if __name__ == '__main__':
    plot_heatmap('FA')
    plot_heatmap('MD')