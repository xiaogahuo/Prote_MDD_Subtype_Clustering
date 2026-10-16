from enigmatoolbox.plotting import plot_subcortical
from enigmatoolbox.utils.useful import reorder_sctx
from enigmatoolbox.datasets import load_example_data
from enigmatoolbox.utils.useful import zscore_matrix
from enigmatoolbox.datasets import load_summary_stats
import pandas as pd

def demo():
    # Load all example data from an individual site
    # cov, metr1_SubVol, metr2_CortThick, metr3_CortSurf = load_example_data()
    cov, metr1_SubVol = pd.read_csv('./cov.csv'), pd.read_csv('./metr1_SubVol.csv')
    metr1_SubVol_r = reorder_sctx(metr1_SubVol)
    group = cov['Dx'].to_list()
    controlCode = 0
    SV_z = zscore_matrix(metr1_SubVol_r.iloc[:, 1:-1], group, controlCode)
    SV_z_mean = SV_z.iloc[cov[cov['SDx'] == 3].index, :].mean(axis=0)
    print(type(SV_z_mean))
    print(SV_z_mean)

    # <class 'pandas.core.series.Series'>
    # Laccumb -56.332102
    # Lamyg -11.948225
    # Lcaud -3.897068
    # Lhippo -3.280988
    # Lpal -14.612755
    # Lput    0.575939
    # Lthal    1.784339
    # LLatVent    1.442127
    # Raccumb -46.921847
    # Ramyg -11.055219
    # Rcaud -3.177580
    # Rhippo -1.076487
    # Rpal -13.769859
    # Rput    0.472401
    # Rthal    2.399190
    # RLatVent    1.781100
    # dtype: float64
    plot_subcortical(array_name=SV_z_mean, size=(800, 400),
                     cmap='RdBu_r', color_bar=True, color_range=(-56, 56)) # Blues


def draw(d_series, min_v, max_v):
    plot_subcortical(array_name=d_series, size=(700, 400),
                     cmap='RdBu_r', color_bar=True, color_range=(min_v, max_v)) # Blues


if __name__ == '__main__':
    # demo()
    df = pd.read_excel("./1102Subcortical_volumes.xlsx", sheet_name=0, index_col='index_col')
    print(df.columns.tolist())
    clust1_Estimate, clust2_Estimate = df['clust1_Estimate'], df['clust2_Estimate']
    print(len(clust1_Estimate.tolist()), len(clust2_Estimate.tolist()))
    print(clust1_Estimate)
    draw(clust1_Estimate, -0.4, 0.4)
    draw(clust2_Estimate, -0.4, 0.4)
