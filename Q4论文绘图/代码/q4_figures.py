# -*- coding: utf-8 -*-
"""Q4论文绘图：冻结Q3输入核验、分区与资源比较，不重跑优化。"""
from pathlib import Path
import argparse,csv,json,hashlib
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
NAMES=['A机','B机','C机','A电池','B电池','C电池','中继机','中继组件']
FIELDS=['A型运输无人机数','B型运输无人机数','C型运输无人机数','A型电池组数','B型电池组数','C型电池组数','中继无人机数','中继能源组件数']
COLORS=['#4477AA','#CC6677','#228877']

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def n(r,k):
    v=float(r[k])
    if not np.isfinite(v):raise ValueError('非有限值 '+k)
    return v

def verify(root,groups,comparison,manifest):
    for name,digest in manifest['input_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('冻结Q3输入哈希不一致：'+name)
    trips=read(root/'Q3_运输架次.csv');relays=read(root/'Q3_中继架次.csv');comm=read(root/'Q3_通信保障.csv')
    metrics={r['指标']:n(r,'数值') for r in read(root/'Q3_指标汇总.csv')}
    rm={r['中继架次编号']:r for r in relays};inventory=np.array(manifest['inventory'])
    expected={s for unit in manifest['components'] for s in unit};summary={}
    if sorted(n(r,'K') for r in comparison)!=[2,3]:raise ValueError('需唯一的K=2/3比较方案')
    for c in comparison:
        k=int(n(c,'K'));rows=[r for r in groups if int(n(r,'K'))==k]
        sites=[s for r in rows for s in r['服务区列表'].split(',')]
        if len(rows)!=k or set(sites)!=expected or len(sites)!=len(expected):raise ValueError('服务区覆盖/分组错误')
        if len({r['任务组编号'] for r in rows})!=k:raise ValueError('任务组重复')
        assigned=[]
        for r in rows:
            ss=set(r['服务区列表'].split(','))
            for unit in manifest['components']:
                if ss.intersection(unit) and not set(unit)<=ss:raise ValueError('不可拆单元跨组')
            selected=[t for t in trips if set(t['访问服务区顺序'].split(','))<=ss]
            tids={t['架次编号'] for t in selected};assigned+=list(tids)
            relayids={x['中继架次编号'] for x in comm if x['运输架次编号'] in tids and x['保障方式']=='中继'}
            rr=[rm[x] for x in relayids]
            controls={'运输架次数':len(selected),'中继架次数':len(rr),'货箱数':sum(len(t['货箱编号列表'].split(',')) for t in selected),
                '运输能耗kWh':sum(n(t,'架次能耗kWh') for t in selected),'中继能耗kWh':sum(n(t,'架次能耗kWh') for t in rr),
                '运输任务时长s':sum(n(t,'返回O01时刻s')-n(t,'开始时刻s') for t in selected),
                '中继任务时长s':sum(n(t,'返回O01时刻s')-n(t,'开始时刻s') for t in rr)}
            for key,val in controls.items():
                if abs(n(r,key)-val)>.01:raise ValueError('分区继承结果不一致：'+key)
            if abs(n(r,'工作量s')-n(r,'运输任务时长s')-n(r,'中继任务时长s'))>.01:raise ValueError('工作量口径不一致')
        if Counter(assigned)!=Counter(t['架次编号'] for t in trips):raise ValueError('运输架次遗漏或重复')
        config=np.array([sum(n(r,f) for r in rows) for f in FIELDS]);deficit=np.maximum(config-inventory,0);surplus=np.maximum(inventory-config,0)
        for prefix,values in [('配置_',config),('缺口_',deficit),('冗余_',surplus)]:
            if any(abs(n(c,prefix+name)-v)>1e-8 for name,v in zip(NAMES,values)):raise ValueError('资源配置/缺口/冗余不一致')
        work=np.array([n(r,'工作量s') for r in rows]);cv=work.std(ddof=0)/work.mean()
        if abs(cv-n(c,'工作量变异系数'))>1e-8 or sum(deficit)!=n(c,'资源总缺口') or sum(surplus)!=n(c,'资源总冗余'):raise ValueError('比较指标不一致')
        if max(n(r,'货箱数') for r in rows)-min(n(r,'货箱数') for r in rows)!=n(c,'货箱数极差'):raise ValueError('货箱极差不一致')
        summary[str(k)]={'总配置':int(sum(config)),'资源总缺口':int(sum(deficit)),'资源总冗余':int(sum(surplus)),'工作量CV':cv,
            '工作量s':list(work),'货箱数':[int(n(r,'货箱数')) for r in rows],'运输架次数':[int(n(r,'运输架次数')) for r in rows],
            '独立复制中继后的总能耗kWh':sum(n(r,'运输能耗kWh')+n(r,'中继能耗kWh') for r in rows)}
    return summary,metrics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=ROOT/'结果');p.add_argument('--out',type=Path,default=ROOT/'结果'/'Q4论文图')
    p.add_argument('--nodes',type=Path);p.add_argument('--dpi',type=int,default=600);p.add_argument('--font')
    a=p.parse_args();available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ([a.font] if a.font else ['Microsoft YaHei','SimHei','Noto Sans CJK SC','SimSun']) if f in available),None)
    if not font:raise RuntimeError('请安装中文字体或通过--font指定')
    if a.dpi<=0:p.error('dpi必须为正')
    plt.rcParams.update({'font.family':font,'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'path','legend.frameon':False,'savefig.facecolor':'white'})
    groups=read(a.results/'Q4_分区配置.csv');comparison=sorted(read(a.results/'Q4_方案比较.csv'),key=lambda r:n(r,'K'))
    manifest=json.loads((a.results/'Q4_input_manifest.json').read_text(encoding='utf-8-sig'))
    summary,metrics=verify(a.results,groups,comparison,manifest)
    npth=a.nodes or a.results/'Q4_绘图节点.json';nodeinfo=json.loads(npth.read_text(encoding='utf-8-sig'));nodes={r['id']:r for r in nodeinfo['nodes']}
    a.out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        for ext in ('png','pdf','svg'):fig.savefig(a.out/f'{name}.{ext}',dpi=a.dpi,bbox_inches='tight')
        plt.close(fig);print('已生成',name,flush=True)
    def rows(k):return sorted([r for r in groups if int(n(r,'K'))==k],key=lambda r:r['任务组编号'])
    inventory=np.array(manifest['inventory']);peak=np.array(manifest['global_peak'])
    o=nodes['O01'];rad=np.pi/180
    def xy(s):return 6371.0088*(nodes[s]['lon']-o['lon'])*rad*np.cos(o['lat']*rad),6371.0088*(nodes[s]['lat']-o['lat'])*rad

    fig,axs=plt.subplots(1,2,figsize=(12,6.3),layout='constrained')
    for ax,k in zip(axs,[2,3]):
        for i,r in enumerate(rows(k)):
            sites=r['服务区列表'].split(',');xx,yy=zip(*(xy(s) for s in sites));ax.scatter(xx,yy,s=65,color=COLORS[i],marker=['o','s','^'][i],label=r['任务组编号'])
            for s in sites:ax.annotate(s,xy(s),xytext=(5,5),textcoords='offset points',fontsize=8)
        ax.scatter([0],[0],marker='*',s=140,color='#202A33',label='O01');ax.annotate('O01',(0,0),xytext=(5,6),textcoords='offset points',fontsize=8)
        ax.set(title=f'{k}组分区',xlabel='相对O01的东西向距离 / km',ylabel='相对O01的南北向距离 / km');ax.set_aspect('equal');ax.margins(.15);ax.grid(alpha=.16);ax.legend(ncol=k+1,loc='lower left',fontsize=8)
    fig.suptitle('服务区任务分组（颜色表示任务归属，不表示行政边界）');save(fig,'图1_两组三组空间分区')

    fig,axs=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    vmax=max(n(r,f) for r in groups for f in FIELDS)
    for ax,k in zip(axs,[2,3]):
        rr=rows(k);mat=np.array([[n(r,f) for f in FIELDS] for r in rr]);ax.imshow(mat,cmap='Blues',vmin=0,vmax=vmax,aspect='auto')
        ax.set(xticks=range(8),xticklabels=NAMES,yticks=range(k),yticklabels=[r['任务组编号'] for r in rr],title=f'{k}组方案：组内资源配置数量')
        ax.tick_params(axis='x',rotation=45)
        for i in range(k):
            for j in range(8):ax.text(j,i,f'{mat[i,j]:.0f}',ha='center',va='center',color='white' if mat[i,j]>vmax*.55 else '#243341')
    save(fig,'图2_各组资源配置矩阵')

    fig,ax=plt.subplots(figsize=(11,4.6),layout='constrained');x=np.arange(8);w=.24
    for off,c,color in zip([-.25,.25],comparison,COLORS):
        vals=[n(c,'配置_'+name) for name in NAMES];bars=ax.bar(x+off,vals,width=w*1.8,label=f"{int(n(c,'K'))}组总需求",color=color);ax.bar_label(bars,padding=3,fmt='%.0f',fontsize=9)
    ax.plot(x,inventory,'k_',markersize=24,markeredgewidth=2,label='现有库存',linestyle='none')
    ax.set(xticks=x,xticklabels=NAMES,ylabel='资源数量 / 架或组',ylim=(0,max(inventory.max(),max(n(c,'配置_'+name) for c in comparison for name in NAMES))+1.5),title='独立分组的资源需求与现有库存')
    ax.legend(ncol=3);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);save(fig,'图3_资源需求与库存')

    fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,c in zip(axs,comparison):
        deficit=np.array([n(c,'缺口_'+name) for name in NAMES]);surplus=np.array([n(c,'冗余_'+name) for name in NAMES])
        ax.bar(x,deficit,color='#CC6677',label='缺口');ax.bar(x,-surplus,color='#228877',label='富余')
        for i,(d,s) in enumerate(zip(deficit,surplus)):
            if d:ax.text(i,d+.08,f'{d:.0f}',ha='center',fontsize=9)
            if s:ax.text(i,-s-.08,f'{s:.0f}',ha='center',va='top',fontsize=9)
        ax.axhline(0,color='#555555',lw=.8);ax.set(xticks=x,xticklabels=NAMES,ylabel='数量 / 架或组',ylim=(-2.7,3.8),title=f"{int(n(c,'K'))}组：缺口{n(c,'资源总缺口'):.0f}，富余{n(c,'资源总冗余'):.0f}")
        ax.tick_params(axis='x',rotation=45);ax.legend(ncol=2,fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('分资源类型的缺口与富余（不同类型不能互相抵消）');save(fig,'图4_资源缺口与富余')

    fig,axs=plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
    for col,k in enumerate([2,3]):
        rr=rows(k);xx=np.arange(k);names=[r['任务组编号'] for r in rr]
        transport=np.array([n(r,'运输任务时长s')/3600 for r in rr]);relay=np.array([n(r,'中继任务时长s')/3600 for r in rr]);total=transport+relay
        axs[0,col].bar(xx,transport,label='运输任务占用',color='#4477AA');axs[0,col].bar(xx,relay,bottom=transport,label='中继任务占用',color='#D18B27')
        for i,v in enumerate(total):axs[0,col].text(i,v+.1,f'{v:.2f}',ha='center',fontsize=9)
        axs[0,col].axhline(total.mean(),color='#555555',ls='--',lw=1,label='组均值')
        axs[0,col].set(xticks=xx,xticklabels=names,ylabel='累计工作量 / h',ylim=(0,11.5),title=f"{k}组：工作量CV={summary[str(k)]['工作量CV']:.2%}");axs[0,col].legend(fontsize=8,ncol=2)
        for off,key,label,color in [(-.18,'货箱数','货箱数','#228877'),(.18,'运输架次数','运输架次数','#CC6677')]:
            bars=axs[1,col].bar(xx+off,[n(r,key) for r in rr],width=.34,label=label,color=color);axs[1,col].bar_label(bars,fmt='%.0f',padding=3)
        axs[1,col].set(xticks=xx,xticklabels=names,ylabel='箱数 / 架次数',ylim=(0,58));axs[1,col].legend(ncol=2,fontsize=8)
    for ax in axs.flat:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('工作量均衡与货箱、运输架次数分配');save(fig,'图5_工作量与任务均衡')

    fig,axs=plt.subplots(1,2,figsize=(11,4.7),layout='constrained');labels=['未分区Q3','两组独立','三组独立']
    counts=[sum(peak)]+[summary[str(k)]['总配置'] for k in [2,3]]
    bars=axs[0].bar(labels,counts,color=['#89979F']+COLORS[:2]);axs[0].bar_label(bars,fmt='%.0f',padding=4)
    axs[0].set(ylabel='配置总数 / 混合资源单位',ylim=(0,max(counts)*1.2),title='各类资源数量之和（非成本）')
    te=[metrics['运输能耗kWh']]+[sum(n(r,'运输能耗kWh') for r in rows(k)) for k in [2,3]]
    re=[metrics['中继能耗kWh']]+[sum(n(r,'中继能耗kWh') for r in rows(k)) for k in [2,3]]
    axs[1].bar(labels,te,label='运输',color='#4477AA');axs[1].bar(labels,re,bottom=te,label='中继',color='#D18B27')
    for i,v in enumerate(np.array(te)+re):axs[1].text(i,v+.7,f'{v:.2f}',ha='center')
    axs[1].set(ylabel='总能耗 / kWh',ylim=(0,max(np.array(te)+re)*1.18),title='各组独立复制完整中继任务窗口');axs[1].legend(ncol=2)
    for ax in axs:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,'图6_独立分组的资源与能耗代价')
    files=['Q4_分区配置.csv','Q4_方案比较.csv','Q4_input_manifest.json']+list(manifest['input_sha256'])
    report={'repository':'https://github.com/jinding-droid/Drone-transportation','reference_commit':'9b407179971ab49327fc2d23b70ac620ae1c0856',
        'inputs_sha256':{f:hashlib.sha256((a.results/f).read_bytes()).hexdigest() for f in files},'nodes_sha256':hashlib.sha256(npth.read_bytes()).hexdigest(),
        'summary':summary,'balance_cv_limit':manifest['balance_cv_limit'],'relay_policy':manifest['relay_policy'],'font':font,'dpi':a.dpi,
        'limitations':'核验输入哈希、分组继承、工作量及库存算术，未重算含充电的峰值配置或重新证明优化最优性。节点为近似平面示意；资源单位合计非经济成本。'}
    (a.out/'数据核对与来源.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('输出目录',a.out.resolve())

if __name__=='__main__':main()
