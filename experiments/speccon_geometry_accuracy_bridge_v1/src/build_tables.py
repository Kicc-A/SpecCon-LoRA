#!/usr/bin/env python3
"""Write publication table source from completed results only."""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    c=read(ROOT/'results/calibration_ref_bridge.csv')
    p=read(ROOT/'results/global_rank_map_counterfactual.csv')
    assert len(c)==6 and len(p)==3 and all(x['status']=='observed' for x in c+p)
    lines=[r'\begin{table}[t]',r'\centering',
        r'\caption{Controlled MRPC reference-calibration bridge: fixed-final accuracy (\%).}',
        r'\begin{tabular}{rrrr}',r'\hline',
        'Seed & Energy & Full & Full--Energy '+r'\\',r'\hline']
    for seed in (42,777,2026):
        f=next(x for x in c if int(x['seed'])==seed and x['arm']=='F')
        e=next(x for x in c if int(x['seed'])==seed and x['arm']=='E')
        full,energy=float(f['final_accuracy_percent']),float(e['final_accuracy_percent'])
        lines.append(f'{seed} & {energy:.3f} & {full:.3f} & {full-energy:+.3f} '+r'\\')
    lines += [r'\hline',r'\end{tabular}',r'\end{table}','',
        r'\begin{table}[t]',r'\centering',
        r'\caption{CoLA type-matched P1 permutation: fixed-final MCC $\times 100$.}',
        r'\begin{tabular}{rrrr}',r'\hline',
        'Seed & Full & P1 & Full--P1 '+r'\\',r'\hline']
    for x in p:
        lines.append(f"{x['seed']} & {float(x['full_final_mcc_percent']):.3f} & "
            f"{float(x['p1_final_mcc_percent']):.3f} & "
            f"{float(x['delta_full_minus_p1']):+.3f} "+r'\\')
    lines += [r'\hline',r'\end{tabular}',r'\end{table}']
    (ROOT/'analysis/TABLES_FOR_PAPER.tex').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
