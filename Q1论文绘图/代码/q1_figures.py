#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 main 分支发布的 Q1 CSV 绘制论文图；不重跑或改变优化模型。
运行: python 代码/q1_figures.py
依赖: python -m pip install numpy matplotlib
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, PercentFormatter

COLORS = {'A': '#4477AA', 'B': '#228877', 'C': '#CC6677', '混合机型': '#6655AA'}
ROOT = Path(__file__).resolve().parents[1]
FILES = ['Q1_最大安全载荷.csv', 'Q1_机型对比.csv', 'Q1_组批方案.csv',
         'Q1_安全余量灵敏度.csv', 'Q1_全局方案灵敏度.csv']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f'非有限值: {key}={row[key]}')
    return value


def configure(font):
    available = {f.name for f in font_manager.fontManager.ttflist}
    choices = [font] if font else ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'SimSun', 'Arial Unicode MS']
    chosen = next((f for f in choices if f in available), None)
    if not chosen:
        raise RuntimeError('未找到中文字体，请安装思源黑体或使用 --font 指定已安装的中文字体名称。')
    plt.rcParams.update({'font.family': chosen, 'font.size': 10,
        'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titlepad': 12, 'axes.labelpad': 7, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'path', 'legend.frameon': False,
        'savefig.facecolor': 'white'})
    return chosen


def validate(caps, comparison, batches, local, glob):
    sites = [r['服务区编号'] for r in caps]
    if len(sites) != 15 or len(set(sites)) != 15:
        raise ValueError('安全载荷表应含15个不同服务区')
    expected = {(s, g) for s in sites for g in 'ABC'}
    actual = [(r['服务区编号'], r['机型']) for r in comparison]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('机型对比表存在缺失或重复')
    ids = [b.strip() for r in batches for b in r['货箱编号列表'].split(',')]
    if len(ids) != 80 or len(set(ids)) != 80:
        raise ValueError('组批方案必须覆盖80个不重复货箱')
    if set(r['服务区编号'] for r in batches) != set(sites):
        raise ValueError('组批方案服务区不完整')
    if len({r['架次编号'] for r in batches}) != len(batches):
        raise ValueError('重复架次编号')
    capmap = {r['服务区编号']: r for r in caps}
    for r in batches:
        if any(not b.startswith(r['服务区编号'] + '-') for b in r['货箱编号列表'].split(',')):
            raise ValueError('第一问不得跨服务区组批')
        if number(r, '总质量kg') > number(capmap[r['服务区编号']], r['机型编号']+'最大安全载荷kg') + .002:
            raise ValueError('载质量超过安全载荷')
        if not 20 - .011 <= number(r, '返航SOC%') <= 100:
            raise ValueError('20%基准安全余量未满足')
    weight = sum(number(r, '总质量kg') for r in batches)
    volume = sum(number(r, '总体积m³') for r in batches)
    energy = sum(number(r, '架次能耗kWh') for r in batches)
    time = sum(number(r, '往返时间s') for r in batches)
    if abs(weight-758) > .01 or abs(volume-2.011) > .001:
        raise ValueError('货物总质量或总体积与本题不符')
    base = [r for r in glob if abs(number(r, '返航安全余量rho')-.2) < 1e-8]
    if len(base) != 1:
        raise ValueError('全局灵敏度表缺少唯一rho=0.20基准')
    base = base[0]
    # 逐行CSV采用4位能耗、1位时间的小数舍入，求和与整体舍入允许其误差上界。
    if (number(base, '架次数') != len(batches) or abs(number(base, '总能耗kWh')-energy) > (len(batches)+1)*.000051
            or abs(number(base, '累计作业时间s')-time) > (len(batches)+1)*.051):
        raise ValueError('组批方案和全局灵敏度基准不一致，可能混用了版本')
    counts = Counter(r['机型编号'] for r in batches)
    if any(counts[g] != number(base, g+'架次') for g in 'ABC'):
        raise ValueError('机型架次数与基准不一致')
    for s in {r['服务区编号'] for r in local}:
        for g in 'ABC':
            rows = sorted((r for r in local if r['服务区编号']==s and r['机型']==g), key=lambda r:number(r,'返航安全余量rho'))
            q = [number(r, '最大安全载荷kg') for r in rows]
            if any(b > a+.002 for a,b in zip(q,q[1:])):
                raise ValueError('安全余量增加而安全载荷上升，请检查源表')
    return dict(sorties=len(batches), boxes=len(ids), mass_kg=weight, volume_m3=volume,
        model_counts=dict(counts), energy_sum_rounded_rows_kwh=energy,
        cumulative_time_sum_rounded_rows_s=time, published_energy_kwh=number(base,'总能耗kWh'),
        published_cumulative_time_s=number(base,'累计作业时间s'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT/'结果')
    p.add_argument('--out', type=Path, default=ROOT/'结果'/'Q1论文图')
    p.add_argument('--dpi', type=int, default=600)
    p.add_argument('--font', help='已安装的中文字体名称')
    args = p.parse_args()
    if args.dpi <= 0:
        p.error('--dpi必须为正数')
    font = configure(args.font)
    caps, comp, batches, local, glob = [read_csv(args.results/f) for f in FILES]
    summary = validate(caps, comp, batches, local, glob)
    args.out.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(args.out/f'{name}.{ext}', dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print('已生成', name, flush=True)
    sites = sorted(r['服务区编号'] for r in caps)
    capmap = {r['服务区编号']:r for r in caps}

    # 图1：绝对安全载荷；数值为能量约束反解上限，不是实际架次装载量。
    mat = np.array([[number(capmap[s],g+'最大安全载荷kg') for g in 'ABC'] for s in sites])
    fig, ax = plt.subplots(figsize=(6.4, 7.2), layout='constrained')
    im = ax.imshow(mat, cmap='YlGnBu', vmin=0, vmax=max(80,float(mat.max())), aspect='auto')
    ax.set(xticks=range(3), xticklabels=['A型','B型','C型'], yticks=range(len(sites)), yticklabels=sites,
           title='各服务区的最大安全载荷（安全余量20%）')
    for i in range(len(sites)):
        for j in range(3):
            ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',color='white' if mat[i,j]>50 else '#152A35')
    fig.colorbar(im,ax=ax,label='最大安全载荷 / kg',shrink=.8)
    save(fig,'图1_最大安全载荷')

    # 图2：三个独立坐标轴，避免双轴与不同单位直接归一化造成误读。
    labels = ['A型','B型','C型','混合机型']
    metrics = [('架次数','架次数 / 次',1),('总能耗kWh','总能耗 / kWh',1),('总作业时间s','累计作业时间 / h',3600)]
    fig, axs = plt.subplots(1,3,figsize=(12,3.9),layout='constrained')
    for ax,(key,ylabel,div) in zip(axs,metrics):
        vals = [sum(number(r,key) for r in comp if r['机型']==g)/div for g in 'ABC']
        vals += [({'架次数':summary['sorties'],'总能耗kWh':summary['published_energy_kwh'],
                   '总作业时间s':summary['published_cumulative_time_s']}[key])/div]
        bars=ax.bar(labels,vals,color=list(COLORS.values()),width=.63)
        ax.bar_label(bars,labels=[f'{v:.0f}' if key=='架次数' else f'{v:.2f}' for v in vals],padding=4)
        ax.set(ylabel=ylabel,ylim=(0,max(vals)*1.18)); ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('统一机型与混合机型方案比较（均完成全部货物运输）')
    save(fig,'图2_机型方案比较')

    fig,ax=plt.subplots(figsize=(10,4.1),layout='constrained')
    bottom=np.zeros(len(sites))
    for g in 'ABC':
        v=np.array([sum(r['服务区编号']==s and r['机型编号']==g for r in batches) for s in sites])
        ax.bar(sites,v,bottom=bottom,label=g+'型',color=COLORS[g],width=.65);bottom+=v
    ax.yaxis.set_major_locator(MaxNLocator(integer=True)); ax.set(ylabel='运输架次数 / 次',ylim=(0,max(bottom)+.65),title='混合机型方案的服务区架次分配')
    ax.tick_params(axis='x',rotation=45);ax.legend(ncol=3,loc='upper right');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,'图3_服务区机型分配')

    fig,axs=plt.subplots(2,1,figsize=(11,6.5),sharex=True,layout='constrained')
    x=np.arange(len(batches)); colors=[COLORS[r['机型编号']] for r in batches]
    util=[number(r,'总质量kg')/number(capmap[r['服务区编号']],r['机型编号']+'最大安全载荷kg')*100 for r in batches]
    axs[0].bar(x,util,color=colors);axs[0].axhline(100,color='#555555',ls='--',lw=1)
    axs[0].set(ylabel='安全载荷利用率 / %',ylim=(0,115),title='各架次的安全载荷利用率与返航电量')
    soc=[number(r,'返航SOC%') for r in batches]
    axs[1].bar(x,soc,color=colors);axs[1].axhline(20,color='#B53A3A',ls='--',label='最低返航SOC：20%')
    axs[1].set(ylabel='返航SOC / %',ylim=(0,100),xticks=x,
        xticklabels=[r['架次编号']+'\n'+r['服务区编号']+'\n'+r['机型编号'] for r in batches])
    axs[1].tick_params(axis='x',labelsize=8);axs[1].legend(loc='upper right')
    for ax in axs:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,'图4_架次利用率与返航SOC')

    ls=sorted({r['服务区编号'] for r in local})
    fig,axs=plt.subplots(math.ceil(len(ls)/2),2,figsize=(10,6.7),sharex=True,sharey=True,layout='constrained',squeeze=False)
    for ax,s in zip(axs.flat,ls):
        for g in 'ABC':
            rows=sorted([r for r in local if r['服务区编号']==s and r['机型']==g],key=lambda r:number(r,'返航安全余量rho'))
            ax.plot([number(r,'返航安全余量rho') for r in rows],[number(r,'最大安全载荷kg') for r in rows],
                marker={'A':'o','B':'s','C':'^'}[g],ms=4,color=COLORS[g],label=g+'型')
        ax.axvline(.2,color='#888888',ls=':',lw=1);ax.set(title=s,xlabel='返航安全余量',ylabel='最大安全载荷 / kg',ylim=(0,85))
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.18)
    for ax in list(axs.flat)[len(ls):]:ax.set_visible(False)
    axs.flat[0].legend(ncol=3);fig.suptitle('典型服务区的安全余量敏感性（曲线连接离散计算点）')
    save(fig,'图5_安全载荷敏感性')

    glob=sorted(glob,key=lambda r:number(r,'返航安全余量rho'))
    rho=np.array([number(r,'返航安全余量rho') for r in glob])
    fig,axs=plt.subplots(1,3,figsize=(12,4.1),layout='constrained')
    for ax,key,label,div in zip(axs,['架次数','总能耗kWh','累计作业时间s'],['架次数 / 次','总能耗 / kWh','累计作业时间 / h'],[1,1,3600]):
        y=[number(r,key)/div if r['可行性']=='可行' else np.nan for r in glob]
        ax.plot(rho,y,'o-',color=COLORS['混合机型'],ms=5)
        ax.axvline(.2,color='#888888',ls=':',lw=1)
        for r in glob:
            if r['可行性']!='可行':
                xx=number(r,'返航安全余量rho');ax.axvline(xx,color='#B53A3A',ls='--',lw=1)
                ax.text(xx,.9,'不可行',transform=ax.get_xaxis_transform(),ha='right',color='#B53A3A',fontsize=9)
        ax.set(xlabel='返航安全余量',ylabel=label,xlim=(rho.min()-.01,rho.max()+.025))
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.18)
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle('混合机型方案随安全余量的变化（离散情景比较）')
    save(fig,'图6_全局方案敏感性')
    manifest={'source_repository':'https://github.com/jinding-droid/Drone-transportation',
        'reference_commit':'9b407179971ab49327fc2d23b70ac620ae1c0856',
        'note':'reference_commit为交付时的参考版本；本次实际输入以以下SHA256为准。只核对表间一致性，未重新证明物理模型。',
        'inputs':{f:{'path':str((args.results/f).resolve()),'sha256':hashlib.sha256((args.results/f).read_bytes()).hexdigest()} for f in FILES},
        'font':font,'dpi':args.dpi,'validation':summary}
    (args.out/'数据核对与来源.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('输出目录:',args.out.resolve())


if __name__=='__main__':
    main()
