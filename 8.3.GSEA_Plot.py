from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
from SUtils import *


def plot_A(input, outdir):
    # 读取数据
    df = pd.read_excel(input, sheet_name="Panel_A")
    df.columns = df.columns.str.strip()
    df = df.sort_values("Order", ascending=True).copy()

    cols = ["NES_Discovery", "NES_Test"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    fig, ax = plt.subplots(figsize=(8, 8))
    y = np.arange(len(df))

    # 纵向网格线及 NES=0 参考线
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.9)
    ax.axvline(0, color="#C8C8C8", linewidth=1.1, zorder=1)

    # Discovery：蓝色圆点
    ax.scatter(df["NES_Discovery"], y, s=95, marker="o", color="#245680", edgecolors="white",
        linewidths=0.7,label="Discovery",zorder=3,)

    # Test：橙色方块，绘制在圆点上层，与参考图一致
    ax.scatter(df["NES_Test"], y,  s=95, marker="s", color="#E87928", edgecolors="white",
        linewidths=0.7, label="Test", zorder=4,)

    # 坐标轴
    ax.set_yticks(y)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_ylim(len(df) - 0.7, -0.3)  # 第一条通路位于顶部

    ax.set_xlim(-2.55, 2.0)
    ax.set_xticks([-2, -1, 0, 1])
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", labelsize=11, length=4)

    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_color("#404040")
    ax.spines["bottom"].set_linewidth(1.2)

    # 图例放在底部
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False,
        fontsize=12, handletextpad=0.5, columnspacing=2.2)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.15)
    fig.savefig(outdir, dpi=600, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def plot_B(input, outdir):
    df = pd.read_excel(input, sheet_name="Panel_B")
    df.columns = df.columns.str.strip()
    df = df.sort_values("Order", ascending=True).copy()

    nes_cols = ["NES_Subtype1", "NES_Subtype2"]
    df[nes_cols] = df[nes_cols].apply(pd.to_numeric, errors="raise")

    Subtpye1_color = "#F79F1F"
    Subtpye2_color = "#EA2027"

    y = np.arange(len(df))
    bar_height = 0.34
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.invert_yaxis()
    ax.barh(
        y - bar_height / 2,
        df["NES_Subtype1"],
        height=bar_height,
        color=Subtpye1_color,
        label="Subtype 1 vs HC",
        zorder=3,
    )

    ax.barh(
        y + bar_height / 2,
        df["NES_Subtype2"],
        height=bar_height,
        color=Subtpye2_color,
        label="Subtype 2 vs HC",
        zorder=3,
    )

    # 对称坐标范围，确保所有条形完整显示
    max_abs = np.abs(df[nes_cols].to_numpy()).max()
    limit = np.ceil(max_abs * 1.10 * 2) / 2
    ax.set_xlim(-limit, limit)
    ax.xaxis.set_major_locator(MultipleLocator(1))

    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.axvline(0, color="#A6A6A6", linewidth=1, zorder=2)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="x", length=4)
    ax.tick_params(axis="y", left=False, labelleft=False)

    # ---------- 4. 图例与导出 ----------
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=2,
        columnspacing=2,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.15)
    fig.savefig(outdir, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_C(input, outdir):
    df = pd.read_excel(input, sheet_name="Panel_C_counts")
    data = df[["GO", "KEGG", "Reactome"]].apply(pd.to_numeric, errors="raise")

    cmap = LinearSegmentedColormap.from_list("custom_blues",
        [(0.00, "#F4F7FB"), (0.15, "#DCE9F5"), (0.83, "#3C6B99"), (1.00, "#19385C")])
    sns.set_theme(style="white")

    # 3. 绘制热力图
    fig, ax = plt.subplots(figsize=(3, 3))

    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=data.to_numpy().max(),
        annot=True,
        fmt="d",
        annot_kws={"fontsize": 8},
        linewidths=1,
        linecolor="white",
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )

    # 深色单元格用白字，其余用深色字
    threshold = data.to_numpy().max() * 0.6
    for text, value in zip(ax.texts, data.to_numpy().ravel()):
        text.set_color("white" if value > threshold else "#233342")

    # 隐藏轴标题、刻度和边框
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(
        left=False, bottom=False,
        labelleft=False, labelbottom=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0.2)

    # 4. 保存图片
    fig.savefig(outdir, dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.show()
    plt.close(fig)


if __name__=='__main__':
    input = result_path + "GSEA_figure_plotting_data.xlsx"
    output_A = result_path + 'GSEA_Plot_A.png'
    output_B = result_path + 'GSEA_Plot_B.png'
    output_C = result_path + 'GSEA_Plot_C.png'
    plot_A(input=input, outdir=output_A)
    plot_B(input=input, outdir=output_B)
    plot_C(input=input, outdir=output_C)