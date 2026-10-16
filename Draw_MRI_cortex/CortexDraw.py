from enigmatoolbox.datasets import load_summary_stats
from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical
from enigmatoolbox.datasets import load_example_data
import pandas as pd
import math

# conda activate junyang

def demo():

    # # Load summary statistics for ENIGMA-Autism
    # sum_stats = load_summary_stats('asd')
    # # Get case-control cortical thickness table
    # CT = sum_stats['CortThick_case_vs_controls_meta_analysis']
    CT = pd.read_csv("./asd_meta-analysis_case-controls_CortThick.csv")
    CT = CT[['Structure', 'd_icv']]
    print(CT.columns.tolist())
    # CT.to_csv("./Cortex_Data_Demo.csv", index=None)

    # Extract Cohen's d values
    CT_d = CT['d_icv']
    print(type(CT_d))
    print(CT_d)

    # Map parcellated data to the surface
    CT_d_fsa5 = parcel_to_surface(CT_d, 'aparc_fsa5', fill=0) # fill NA to 100
    print(CT_d_fsa5)

    # Project the results on the surface brain
    plot_cortical(array_name=CT_d_fsa5, surface_name="fsa5", size=(800, 400),
                  cmap='RdBu_r', color_bar=True, color_range=(-0.5, 0.5))
# cmap =
# 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r',
# 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges',
# 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r',
# 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r',
# 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
# 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu',
# 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary',
# 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm',
# 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r',
# 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r',
# 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray',
# 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral',
# 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
# 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r',
# 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r',
# 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
def draw(f_name, sheet_name, f_result):
    df = pd.read_excel("./{}.xlsx".format(f_name), sheet_name=sheet_name)
    print(df.columns.tolist())
    clust1_Estimate, clust2_Estimate = df['clust1_Estimate'], df['clust2_Estimate']
    print(len(clust1_Estimate.tolist()), len(clust2_Estimate.tolist()))
    # f_SB_list = list(filter(lambda x: not math.isnan(x), SB_series.tolist()))
    # f_SD_list = list(filter(lambda x: not math.isnan(x), SD_series.tolist()))
    print("f_name: {}, sheet_name: {}, min: {}, min: {}, max: {}, max: {}".format(f_name, sheet_name,
        min(clust1_Estimate), min(clust2_Estimate), max(clust1_Estimate), max(clust2_Estimate)))
    min_v, max_v = -0.4, 0.4
    CT_d_fsa5 = parcel_to_surface(clust1_Estimate, 'aparc_fsa5')
    filename = "./{}_{}_{}.png".format(f_name, f_result, 'clust1')
    plot_cortical(array_name=CT_d_fsa5, surface_name="fsa5", size=(700, 400),
                  cmap='RdBu_r', color_bar=True, color_range=(min_v, max_v), filename=filename) # cmap = 'RdBu_r'

    CT_d_fsa5 = parcel_to_surface(clust2_Estimate, 'aparc_fsa5')
    plot_cortical(array_name=CT_d_fsa5, surface_name="fsa5", size=(700, 400),
                  cmap='RdBu_r', color_bar=True, color_range=(min_v, max_v))  # cmap = 'RdBu_r'


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # demo()
    draw("RESULT_mri", 0, "Area")
    draw("RESULT_mri", 1, "Thickness")
    draw("RESULT_mri", 2, "Volume")





