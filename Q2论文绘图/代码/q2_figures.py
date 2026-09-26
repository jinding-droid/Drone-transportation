# -*- coding: utf-8 -*-
"""读取已发布Q2 CSV生成论文图。python 代码/q2_figures.py"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
FILES = ['Q2_运输架次.csv','Q2_逐箱交付.csv','Q2_电池资源台账.csv','Q2_指标汇总.csv','Q2_方案对比.csv']
C = {'A':'#4477AA','B':'#228877','C':'#CC6677'}


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def n(row,key):
    v=float(row[key])
    if not np.isfinite(v):raise ValueError(f'非有限值 {key}')
    return v


def validate(trips,boxes,bats,metrics,comparison):
    ids=[b.strip() for r in trips for b in r['货箱编号列表'].split(',')]
    delivered=[r['货箱编号'] for r in boxes]
    if len(ids)!=80 or len(set(ids))!=80 or len(delivered)!=80 or set(ids)!=set(delivered):
        raise ValueError('80箱覆盖或唯一性不满足')
    tm={r['架次编号']:r for r in trips}
    if len(tm)!=len(trips):raise ValueError('架次号重复')
    if Counter(r['架次编号'] for r in bats)!=Counter(tm.keys()):raise ValueError('电池台账架次缺失/重复')
    for r in trips:
        if n(r,'开始时刻s')<0 or n(r,'返回O01时刻s')<n(r,'开始时刻s'):raise ValueError('非法任务时间')
        if not 20-.001<=n(r,'返航SOC%')<=100:raise ValueError('SOC不满足20%安全余量')
        if n(r,'架次能耗kWh')<=0:raise ValueError('非法能耗')
    for r in boxes:
        t=tm[r['架次编号']]; when=n(r,'交付完成时刻s')
        if r['货箱编号'] not in t['货箱编号列表'].split(','):raise ValueError('交付架次映射错误')
        if r['服务区编号'] not in t['访问服务区顺序'].split(','):raise ValueError('服务区映射错误')
        if not n(t,'开始时刻s')-.002<=when<=n(t,'返回O01时刻s')+.002:raise ValueError('交付不在任务窗口内')
    for r in bats:
        t=tm[r['架次编号']]
        if r['电池编号']!=t['电池编号'] or r['机型编号']!=t['机型编号']:raise ValueError('电池映射错误')
        for a,b in [('占用开始s','开始时刻s'),('任务返回s','返回O01时刻s'),('返航SOC%','返航SOC%')]:
            if abs(n(r,a)-n(t,b))>.002:raise ValueError('电池台账与任务不一致')
        if n(r,'充电完成s')<n(r,'任务返回s'):raise ValueError('充电结束早于返场')
    for rows,idkey,start,end in [(trips,'无人机编号','开始时刻s','返回O01时刻s'),(bats,'电池编号','占用开始s','充电完成s')]:
        for ident in {r[idkey] for r in rows}:
            seq=sorted([r for r in rows if r[idkey]==ident],key=lambda r:n(r,start))
            if any(n(b,start)<n(a,end)-.002 for a,b in zip(seq,seq[1:])):raise ValueError(f'{ident}占用冲突')
    hard=sum(bool(r['硬时限s']) and n(r,'交付完成时刻s')>n(r,'硬时限s')+.002 for r in boxes)
    late=sum(n(r,'交付完成时刻s')>n(r,'期望送达时刻s')+.002 for r in boxes)
    m={r['指标']:n(r,'数值') for r in metrics}
    energy=sum(n(r,'架次能耗kWh') for r in trips); finish=max(n(r,'返回O01时刻s') for r in trips)
    if abs(energy-m['energy_kwh'])>.0001 or abs(finish-m['makespan_s'])>.002 or len(trips)!=m['sorties'] or hard!=m['hard_late_count']:
        raise ValueError('指标汇总与明细不一致')
    if m['soft_late_count']==0 and late!=0:raise ValueError('零迟到汇总与明细不符')
    main=comparison[0]
    if abs(n(main,'任务完成时间s')-finish)>.002 or abs(n(main,'运输能耗kWh')-energy)>.0001 or n(main,'架次数')!=len(trips):
        raise ValueError('方案对比首行与发布方案不同')
    return dict(sorties=len(trips),boxes=len(boxes),models=dict(Counter(r['机型编号'] for r in trips)),
        drones=len({r['无人机编号'] for r in trips}),batteries=len({r['电池编号'] for r in trips}),
        multi_stop_sorties=sum(',' in r['访问服务区顺序'] for r in trips),hard_late=hard,all_expected_late=late,
        published_makespan_s=m['makespan_s'],published_energy_kwh=m['energy_kwh'],
        last_delivery_s=max(n(r,'交付完成时刻s') for r in boxes),last_charge_s=max(n(r,'充电完成s') for r in bats))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'结果');p.add_argument('--out',type=Path,default=ROOT/'结果'/'Q2论文图')
    p.add_argument('--dpi',type=int,default=600);p.add_argument('--font',default=None)
    a=p.parse_args()
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ([a.font] if a.font else ['Microsoft YaHei','SimHei','Noto Sans CJK SC','SimSun']) if f in fonts),None)
    if not font:raise RuntimeError('请安装中文字体或用 --font 指定中文字体名称')
    if a.dpi<=0:p.error('dpi必须为正')
    plt.rcParams.update({'font.family':font,'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'path','legend.frameon':False,'savefig.facecolor':'white'})
    trips,boxes,bats,metrics,comparison=[read(a.results/f) for f in FILES]
    summary=validate(trips,boxes,bats,metrics,comparison)
    a.out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        for ext in ('png','pdf','svg'):fig.savefig(a.out/f'{name}.{ext}',dpi=a.dpi,bbox_inches='tight')
        plt.close(fig);print('已生成',name,flush=True)
    finish=summary['published_makespan_s']/60

    fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
    drones=sorted({r['无人机编号'] for r in trips})
    for r in trips:
        y=drones.index(r['无人机编号']);lo=n(r,'开始时刻s')/60;hi=n(r,'返回O01时刻s')/60
        ax.barh(y,hi-lo,left=lo,height=.64,color=C[r['机型编号']],edgecolor='white')
        label=r['架次编号'].replace('Q2-','')+'\n'+'→'.join(s.replace('S00','S').replace('S0','S') for s in r['访问服务区顺序'].split(','))
        ax.text((lo+hi)/2,y,label,ha='center',va='center',fontsize=8,color='white')
    ax.set(yticks=range(len(drones)),yticklabels=[d+'（'+next(r['机型编号'] for r in trips if r['无人机编号']==d)+'型）' for d in drones],
        xlabel='任务开始后的时间 / min',title='运输无人机任务占用甘特图',xlim=(0,finish+4))
    ax.invert_yaxis();ax.axvline(finish,color='#555555',ls='--',lw=1)
    ax.legend(handles=[Patch(color=C[g],label=g+'型') for g in 'ABC'],ncol=3,loc='upper left',bbox_to_anchor=(0,1.12))
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);save(fig,'图1_无人机调度甘特图')

    fig,ax=plt.subplots(figsize=(12,8),layout='constrained')
    batteries=sorted({r['电池编号'] for r in bats})
    for r in bats:
        y=batteries.index(r['电池编号']);lo=n(r,'占用开始s')/60;ret=n(r,'任务返回s')/60;ready=n(r,'充电完成s')/60
        ax.barh(y,ret-lo,left=lo,height=.65,color=C[r['机型编号']])
        ax.barh(y,ready-ret,left=ret,height=.65,color='#D9DDE2',hatch='///',edgecolor='#8C939B',linewidth=.4)
        ax.text((lo+ret)/2,y,r['架次编号'].replace('Q2-',''),ha='center',va='center',color='white',fontsize=8)
    ax.set(yticks=range(len(batteries)),yticklabels=batteries,xlabel='任务开始后的时间 / min',title='电池任务占用与充电周转',xlim=(0,summary['last_charge_s']/60+3))
    ax.invert_yaxis();ax.axvline(finish,color='#555555',ls='--',lw=1,label='运输任务全部返场')
    ax.legend(handles=[Patch(color=C[g],label=g+'型任务占用') for g in 'ABC']+[Patch(facecolor='#D9DDE2',hatch='///',label='充电')],ncol=4,loc='upper left',bbox_to_anchor=(0,1.08))
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);save(fig,'图2_电池周转甘特图')

    fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,key,title in zip(axs,['硬时限s','期望送达时刻s'],['有硬时限货箱','全部货箱的期望送达时刻']):
        rows=[r for r in boxes if r[key]]
        pts=Counter((n(r,key)/60,n(r,'交付完成时刻s')/60) for r in rows)
        lim=max(max(x,y) for x,y in pts)*1.08
        for (x,y),cnt in pts.items():
            ax.scatter(x,y,s=28+cnt*16,color='#4477AA',alpha=.65,edgecolor='white',linewidth=.5)
        ax.plot([0,lim],[0,lim],'--',color='#B53A3A',lw=1,label='交付时刻 = 时限')
        ax.set(xlabel='时限 / min',ylabel='实际交付完成时刻 / min',title=f'{title}（{len(rows)}箱）',xlim=(0,lim),ylim=(0,lim))
        ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(loc='upper left',fontsize=8)
    fig.suptitle('交付时效核验（圆点大小表示重合货箱数量）');save(fig,'图3_交付时效')

    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    counts=Counter(n(r,'交付完成时刻s')/60 for r in boxes);tt=sorted(counts)
    ax.step([0]+tt+[finish],[0]+list(np.cumsum([counts[t] for t in tt]))+[len(boxes)],where='post',lw=2,color='#228877',label='累计已交付货箱')
    ax.axvline(summary['last_delivery_s']/60,color='#228877',ls=':',label=f"最后交付：{summary['last_delivery_s']/60:.2f} min")
    ax.axvline(finish,color='#6655AA',ls='--',label=f'最后返场：{finish:.2f} min')
    ax.set(xlabel='任务开始后的时间 / min',ylabel='累计交付货箱数 / 箱',title='货物交付进度与任务完成时刻',ylim=(0,85),xlim=(0,finish+5))
    ax.legend(loc='upper left');ax.grid(alpha=.18);save(fig,'图4_累计交付进度')

    fig,axs=plt.subplots(1,3,figsize=(12,4.1),layout='constrained')
    for ax,key,ylabel in zip(axs,['count','energy','work'],['运输架次数 / 次','运输能耗 / kWh','无人机累计占用时间 / h']):
        vals=[]
        for g in 'ABC':
            rows=[r for r in trips if r['机型编号']==g]
            vals.append(len(rows) if key=='count' else sum(n(r,'架次能耗kWh') if key=='energy' else (n(r,'返回O01时刻s')-n(r,'开始时刻s'))/3600 for r in rows))
        bars=ax.bar(['A型','B型','C型'],vals,color=list(C.values()),width=.6)
        ax.bar_label(bars,labels=[f'{v:.0f}' if key=='count' else f'{v:.2f}' for v in vals],padding=4)
        ax.set(ylabel=ylabel,ylim=(0,max(vals)*1.2));ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('各机型的任务承担与资源消耗');save(fig,'图5_机型工作量与能耗')

    fig,axs=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.3,1]},layout='constrained')
    names=[r['方案'].replace('方案（存档）','\n（存档）').replace('22架次','22架次\n') for r in comparison]
    vals=[n(r,'任务完成时间s')/60 for r in comparison];colors=['#6655AA']+['#779AAF']*(len(vals)-1)
    bars=axs[0].barh(range(len(vals)),vals,color=colors,height=.6)
    axs[0].bar_label(bars,labels=[f'{v:.2f}' for v in vals],padding=4,fontsize=9)
    axs[0].set(yticks=range(len(vals)),yticklabels=names,xlabel='最晚返场时间 / min',xlim=(0,max(vals)*1.18));axs[0].invert_yaxis()
    for i,r in enumerate(comparison):
        energy=n(r,'运输能耗kWh');time=n(r,'任务完成时间s')/60
        axs[1].scatter(time,energy,color=colors[i],marker='*' if i==0 else 'o',s=150 if i==0 else 65)
        axs[1].annotate(str(i+1),(time,energy),xytext=(6,5),textcoords='offset points')
    axs[0].set_yticklabels([f'{i+1}. {name}' for i,name in enumerate(names)])
    axs[1].set(xlabel='最晚返场时间 / min',ylabel='运输能耗 / kWh',title='编号对应左图；散点为已发布方案')
    for ax in axs:ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('已发布方案的时效与能耗比较（少架次方案含11箱迟到）');save(fig,'图6_方案比较')
    manifest={'repository':'https://github.com/jinding-droid/Drone-transportation','reference_commit':'9b407179971ab49327fc2d23b70ac620ae1c0856',
        'note':'参考提交为交付版本，实际输入以下列哈希为准；校验为CSV一致性和资源时间互斥检查，未重算物理模型。',
        'inputs':{f:hashlib.sha256((a.results/f).read_bytes()).hexdigest() for f in FILES},'summary':summary,'font':font,'dpi':a.dpi}
    (a.out/'数据核对与来源.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('输出目录',a.out.resolve())


if __name__=='__main__':main()
