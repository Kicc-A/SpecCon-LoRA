#!/usr/bin/env python3
"""Build figures from observed bridge data only."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/FIGURES'
OUT.mkdir(parents=True,exist_ok=True)


def read(name):
    with (ROOT/'results'/name).open() as f:return list(csv.DictReader(f))


def make():
    fixed=read('geometry_fixed_map_scores.csv')
    fig,axes=plt.subplots(1,3,figsize=(10,3),sharey=True)
    for ax,task in zip(axes,('cola','mrpc','rte')):
        for i,other in enumerate(('S1','E','P1')):
            values=[]
            for probe in ('B1','B2'):
                f=next(x for x in fixed if x['task']==task and x['map']=='F' and x['probe']==probe)
                o=next(x for x in fixed if x['task']==task and x['map']==other and x['probe']==probe)
                values.append(float(f['q_geom'])-float(o['q_geom']))
            ax.plot([i-.08,i+.08],values,marker='o',label='B1/B2' if i==0 else None)
            ax.plot([i-.08,i+.08],values,color='0.6',lw=.7)
        ax.axhline(0,color='black',lw=.6)
        ax.set_xticks(range(3),['F−S1','F−E','F−P1'])
        ax.set_title(task.upper())
        ax.tick_params(axis='x',rotation=30)
    axes[0].set_ylabel('ΔQ_geom on independent train probes')
    fig.tight_layout();fig.savefig(OUT/'fixed_map_geometry.pdf');plt.close(fig)

    paired=read('paired_geometry_accuracy.csv')
    fig,ax=plt.subplots(figsize=(4.5,3.5))
    for seed in (42,777,2026):
        x=[r for r in paired if r['task']=='cola' and r['contrast']=='F-P1' and int(r['seed'])==seed]
        q=sum(float(r['delta_q_geom']) for r in x)/2
        d=float(x[0]['delta_dev_metric'])
        ax.scatter(q,d,s=48);ax.annotate(str(seed),(q,d),xytext=(5,4),textcoords='offset points')
    ax.axhline(0,color='black',lw=.7);ax.axvline(0,color='black',lw=.7)
    ax.set_xlabel('F−P1 mean ΔQ_geom (B1/B2), historical maps')
    ax.set_ylabel('F−P1 final CoLA ΔMCC ×100')
    fig.tight_layout();fig.savefig(OUT/'cola_p1_geometry_vs_accuracy.pdf');plt.close(fig)

    fig,ax=plt.subplots(figsize=(4.5,4))
    for task,mark in [('cola','o'),('mrpc','s'),('rte','^')]:
        xs=[];ys=[]
        for arm in ('F','E','S1','P1'):
            xs.append(float(next(x for x in fixed if x['task']==task and x['map']==arm and x['probe']=='B1')['q_geom']))
            ys.append(float(next(x for x in fixed if x['task']==task and x['map']==arm and x['probe']=='B2')['q_geom']))
        ax.scatter(xs,ys,marker=mark,label=task.upper())
    lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1])
    ax.plot([lo,hi],[lo,hi],color='0.5',lw=.7,ls='--')
    ax.set_xlabel('Q_geom on B1');ax.set_ylabel('Q_geom on B2');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(OUT/'fixed_map_B1_B2_consistency.pdf');plt.close(fig)


if __name__=='__main__':make()
