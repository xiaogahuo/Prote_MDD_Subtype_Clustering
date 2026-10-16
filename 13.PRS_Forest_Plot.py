import pandas as pd

from SUtils import *
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

def main(input, outdir):
    df = pd.read_excel(input, sheet_name='Sheet2', header=1).dropna(subset=['OR'])
    assert ((df['95% CI lower'] > 0) & (df['95% CI lower'] <= df['OR']) & (df['OR'] <= df['95% CI upper'])).all()
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
    fig, plot = plt.subplots(figsize=(5.6, 5.2), facecolor='white')
    fig.subplots_adjust(left=.06, right=.97, bottom=.16, top=.98)
    ys = [5.45,4.65,3.45,2.65,1.45,.65]
    gray = '#667586'
    colors = {'Discovery':'#286BA6', 'Test':'#D87536'}
    plot.set_ylim(0,6.1)
    plot.set_yticks([])
    for spine in plot.spines.values(): spine.set_visible(False)
    for lo,hi in [(4.12,5.98),(.12,1.98)]:
        plot.axhspan(lo,hi,color='#F0F5F9',lw=0,zorder=0)
    plot.set_xscale('log')
    plot.set_xlim(.85,1.82)
    plot.axvline(1,ls=(0,(4,3)),color='#8B96A0',lw=1.1,zorder=1)
    plot.spines['bottom'].set_visible(True)
    plot.spines['bottom'].set_color('#9BA8B4')
    plot.xaxis.set_major_locator(FixedLocator([.9,1,1.2,1.4,1.6,1.8]))
    plot.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.1f}'))
    plot.xaxis.set_minor_locator(NullLocator())
    plot.tick_params(axis='x',length=4,color='#9BA8B4',labelsize=10,pad=6)
    plot.set_xlabel('Odds Ratio',fontsize=10,color=gray,labelpad=10)
    for i,(_,r) in enumerate(df.iterrows()):
        y=ys[i]; c=colors[r['Cohort']]
        plot.errorbar(r['OR'],y,xerr=[[r['OR']-r['95% CI lower']],[r['95% CI upper']-r['OR']]],fmt='o' if r['Cohort']=='Discovery' else 's',color=c,ecolor=c,elinewidth=2.2,capsize=4,capthick=1.2,markersize=8,mec='white',mew=1,zorder=3)
    fig.savefig(outdir, dpi=240,bbox_inches='tight',pad_inches=.16)
    plt.show()
    plt.close(fig)

if __name__=='__main__':
    input = result_path+"PRS_森林图.xlsx"
    output = result_path+'PRS_森林图.png'
    main(input=input, outdir=output)

