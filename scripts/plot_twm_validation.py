#!/usr/bin/env python3
"""Render the measured held-out restoration means; no model/data execution."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latest',type=Path,required=True)
    parser.add_argument('--best',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():parser.error('Render in the remote Pod')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    latest=json.loads(args.latest.read_text());best=json.loads(args.best.read_text())
    if latest['phase']!='complete' or best['phase']!='complete':raise ValueError('Completed measurements required')
    la=latest['protocols']['full'];be=best['protocols']['full']
    if la['indices_sha256_by_dataset']!=be['indices_sha256_by_dataset']:raise ValueError('Evaluation windows differ')
    colors=['#9CA9BA','#E5A24B','#3976D6','#2A9B7D']
    labels=['Corrupted input','Last-history hold',f"Latest ({latest['checkpoint_step']:,})",f"Best ({best['checkpoint_step']:,})"]
    modes=['noisy','missing','clean']; names=['Noisy current state','Missing + noisy state','Correct input']
    keys=['wrist_mm','root_deg','finger_deg']; titles=['Wrist position (mm)','Root rotation (deg)','Finger rotation (deg)']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,3,figsize=(12,9))
    for row,mode in enumerate(modes):
        for col,key in enumerate(keys):
            ax=axes[row,col]; l=la['equal_source_mean'][mode];b=be['equal_source_mean'][mode]
            values=[l['baseline_'+key],l['history_hold_'+key],l['restored_'+key],b['restored_'+key]]
            bars=ax.bar(range(4),values,color=colors,width=.67,zorder=3)
            ax.set_xticks([]);ax.set_ylim(0,max(values)*1.25 if max(values) else 1)
            ax.grid(axis='y',color='#E5E9F0',zorder=0)
            ax.set_title(titles[col] if row==0 else '',fontsize=12,pad=12)
            if col==0:ax.set_ylabel(names[row],fontsize=11,labelpad=14)
            for bar,value in zip(bars,values):
                ax.annotate(f'{value:.2f}',(bar.get_x()+bar.get_width()/2,bar.get_height()),
                            xytext=(0,4),textcoords='offset points',ha='center',fontsize=10)
    total=sum(la['windows_by_dataset'].values())
    fig.suptitle('Paused TWM: synthetic restoration on held-out motion',fontsize=17,fontweight='bold',y=.985)
    fig.legend([plt.Rectangle((0,0),1,1,color=c) for c in colors],labels,loc='lower center',ncol=4,
               frameon=False,bbox_to_anchor=(.5,.055),fontsize=11)
    fig.text(.5,.025,f'{total:,} windows per condition | Equal mean of 4 source means | Source-fitted MANO wrist/rotations',ha='center',fontsize=10,color='#46536B')
    fig.text(.5,.005,'GT-based history with synthetic noise; overlapping validation windows; real HaWoR errors and independent sensor accuracy not tested.',ha='center',fontsize=9,color='#596579')
    fig.tight_layout(rect=(0,.11,1,.95))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160,facecolor='white',bbox_inches='tight')
    plt.close(fig)
    print(str(args.output))


if __name__=='__main__':main()
