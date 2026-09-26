# -*- coding: utf-8 -*-
"""Q3论文图：只读发布CSV与节点快照，不修改或重跑优化模型。"""
from pathlib import Path
import argparse,csv,json,hashlib
from collections import Counter,defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
FILES=['Q3_运输架次.csv','Q3_中继架次.csv','Q3_通信保障.csv','Q3_逐箱交付.csv','Q3_指标汇总.csv','Q3_方案对比.csv']
MC={'A':'#4477AA','B':'#228877','C':'#CC6677'}
RC=['#D18B27','#7B61A8','#268DAB']
DIRECT='#B7C3CC'

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def n(r,k):
    v=float(r[k])
    if not np.isfinite(v):raise ValueError('非有限值：'+k)
    return v

def check(trips,relays,comm,boxes,metrics,comparison):
    tm={r['架次编号']:r for r in trips};rm={r['中继架次编号']:r for r in relays}
    if len(tm)!=len(trips) or len(rm)!=len(relays):raise ValueError('重复架次编号')
    ids=[b for r in trips for b in r['货箱编号列表'].split(',')]
    if len(ids)!=80 or len(set(ids))!=80 or Counter(ids)!=Counter(r['货箱编号'] for r in boxes):raise ValueError('80箱覆盖错误')
    for r in trips:
        if n(r,'开始时刻s')<0 or n(r,'返回O01时刻s')<=n(r,'开始时刻s'):raise ValueError('运输时间非法')
        if not 19.999<=n(r,'返航SOC%')<=100:raise ValueError('运输SOC异常')
    for r in relays:
        times=[n(r,k) for k in ['开始时刻s','建链完成时刻s','服务结束时刻s','返回O01时刻s']]
        if min(times)<0 or times!=sorted(times):raise ValueError('中继时间非法')
        if not 19.999<=n(r,'返航SOC%')<=100:raise ValueError('中继SOC异常')
    for rows,idkey,start,end,turn in [(trips,'无人机编号','开始时刻s','返回O01时刻s',0),(relays,'中继无人机编号','开始时刻s','返回O01时刻s',300)]:
        for ident in {r[idkey] for r in rows}:
            rr=sorted([r for r in rows if r[idkey]==ident],key=lambda r:n(r,start))
            if any(n(b,start)<n(a,end)+turn-.002 for a,b in zip(rr,rr[1:])):raise ValueError(ident+'实体占用/周转冲突')
    durations=defaultdict(float)
    if set(r['运输架次编号'] for r in comm)!=set(tm):raise ValueError('通信表缺少运输架次')
    for tid,t in tm.items():
        rows=sorted([r for r in comm if r['运输架次编号']==tid],key=lambda r:n(r,'开始时刻s'))
        for r in rows:
            start,end=n(r,'开始时刻s'),n(r,'结束时刻s')
            if end<start or start<n(t,'开始时刻s')-.002 or end>n(t,'返回O01时刻s')+.002:raise ValueError('通信区间越界')
            if r['保障方式']=='中继':
                relay=rm[r['中继架次编号']]
                if start<n(relay,'建链完成时刻s')-.002 or end>n(relay,'服务结束时刻s')+.002:raise ValueError('中继未在服务窗口内')
                mode=r['中继架次编号']
            elif r['保障方式']=='直连':mode='直连'
            else:raise ValueError('未知保障方式')
            durations[mode]+=end-start
        if any(abs(n(b,'开始时刻s')-n(a,'结束时刻s'))>.002 for a,b in zip(rows,rows[1:])):raise ValueError('导出通信区间存在间隙或重叠')
        if abs(n(rows[-1],'结束时刻s')-n(t,'返回O01时刻s'))>.002:raise ValueError('通信表未延续至返场')
    hard=late=0
    for r in boxes:
        t=tm[r['架次编号']];d=n(r,'交付完成时刻s')
        if r['货箱编号'] not in t['货箱编号列表'].split(',') or r['服务区编号'] not in t['访问服务区顺序'].split(','):raise ValueError('交付映射错误')
        if not n(t,'开始时刻s')-.002<=d<=n(t,'返回O01时刻s')+.002:raise ValueError('交付时间越界')
        hard+=bool(r['硬时限s']) and d>n(r,'硬时限s')+.002
        late+=d>n(r,'期望送达时刻s')+.002
    m={r['指标']:n(r,'数值') for r in metrics}
    energy=sum(n(r,'架次能耗kWh') for r in trips);re=sum(n(r,'架次能耗kWh') for r in relays)
    finish=max(n(r,'返回O01时刻s') for r in trips+relays)
    if abs(energy-m['运输能耗kWh'])>.0001 or abs(re-m['中继能耗kWh'])>.0001 or abs(energy+re-m['总能耗kWh'])>.0002:raise ValueError('能耗汇总不一致')
    if abs(finish-m['联合任务完成时间s'])>.002 or len(trips)!=m['运输架次数'] or len(relays)!=m['中继架次数'] or hard!=m['硬时限违约箱数']:raise ValueError('任务汇总不一致')
    if m['普通物资迟到箱数']==0 and late:raise ValueError('零迟到记录不一致')
    if abs(n(comparison[0],'联合完成时间s')-finish)>.002 or abs(n(comparison[0],'总能耗kWh')-m['总能耗kWh'])>.0002:raise ValueError('对比首行不是当前版本')
    return {'运输架次':len(trips),'中继架次':len(relays),'运输机型架次':dict(Counter(r['机型编号'] for r in trips)),
        '硬时限违约':hard,'全部期望时刻迟到':late,'联合完成时间s':m['联合任务完成时间s'],
        '运输能耗kWh':m['运输能耗kWh'],'中继能耗kWh':m['中继能耗kWh'],'总能耗kWh':m['总能耗kWh'],
        '按运输架次累计的保障时长s':dict(durations),'通信区间数':len(comm)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'结果');p.add_argument('--out',type=Path,default=ROOT/'结果'/'Q3论文图')
    p.add_argument('--nodes',type=Path,help='节点快照JSON，默认结果目录中的Q3_绘图节点.json')
    p.add_argument('--dpi',type=int,default=600);p.add_argument('--font')
    a=p.parse_args();fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ([a.font] if a.font else ['Microsoft YaHei','SimHei','Noto Sans CJK SC','SimSun']) if f in fonts),None)
    if not font:raise RuntimeError('请安装中文字体或通过--font指定')
    if a.dpi<=0:p.error('dpi必须为正')
    plt.rcParams.update({'font.family':font,'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'path','legend.frameon':False,'savefig.facecolor':'white'})
    trips,relays,comm,boxes,metrics,comparison=[read(a.results/f) for f in FILES]
    summary=check(trips,relays,comm,boxes,metrics,comparison)
    nodepath=a.nodes or a.results/'Q3_绘图节点.json';nodeinfo=json.loads(nodepath.read_text(encoding='utf-8'))
    nodes={r['id']:r for r in nodeinfo['nodes']}
    if 'O01' not in nodes or any(s not in nodes for r in trips for s in r['访问服务区顺序'].split(',')):raise ValueError('节点快照缺少任务节点')
    a.out.mkdir(parents=True,exist_ok=True)
    colors={r['中继架次编号']:RC[i%len(RC)] for i,r in enumerate(relays)};colors['直连']=DIRECT
    finish=summary['联合完成时间s']/60
    def save(fig,name):
        for ext in ('png','pdf','svg'):fig.savefig(a.out/f'{name}.{ext}',dpi=a.dpi,bbox_inches='tight')
        plt.close(fig);print('已生成',name,flush=True)

    # 同一时间轴上表现运输实体和中继实体占用；服务窗口叠加亮色。
    drones=sorted({r['无人机编号'] for r in trips});rdrones=sorted({r['中继无人机编号'] for r in relays});labels=drones+rdrones
    fig,ax=plt.subplots(figsize=(12,6.6),layout='constrained')
    for r in trips:
        y=labels.index(r['无人机编号']);lo=n(r,'开始时刻s')/60;hi=n(r,'返回O01时刻s')/60
        ax.barh(y,hi-lo,left=lo,height=.64,color=MC[r['机型编号']]);ax.text((lo+hi)/2,y,r['架次编号'].replace('Q2-',''),ha='center',va='center',color='white',fontsize=8)
    for r in relays:
        y=labels.index(r['中继无人机编号']);lo=n(r,'开始时刻s')/60;hi=n(r,'返回O01时刻s')/60;ready=n(r,'建链完成时刻s')/60;end=n(r,'服务结束时刻s')/60
        ax.barh(y,hi-lo,left=lo,height=.65,color='#DDE1E5');ax.barh(y,end-ready,left=ready,height=.65,color=colors[r['中继架次编号']])
        ax.text((ready+end)/2,y,r['中继架次编号'].replace('Q3-',''),ha='center',va='center',color='white',fontsize=8)
        ax.barh(y,5,left=hi,height=.4,facecolor='none',edgecolor='#888888',hatch='///',linewidth=.5)
    ax.axhline(len(drones)-.5,color='#888888',lw=.7);ax.axvline(finish,color='#555555',ls='--',lw=1)
    ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='任务开始后的时间 / min',title='运输与中继无人机协同调度',xlim=(0,finish+7));ax.invert_yaxis()
    ax.legend(handles=[Patch(color=MC[g],label=g+'型运输') for g in 'ABC']+[Patch(color='#DDE1E5',label='中继非服务阶段'),Patch(facecolor='white',edgecolor='#888888',hatch='///',label='返场后5 min周转')],ncol=5,fontsize=8,loc='upper left',bbox_to_anchor=(0,1.1))
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);save(fig,'图1_运输中继协同甘特图')

    order=sorted(trips,key=lambda r:(n(r,'开始时刻s'),r['架次编号']));tids=[r['架次编号'] for r in order]
    fig,ax=plt.subplots(figsize=(12,8.8),layout='constrained')
    for r in comm:
        lo=n(r,'开始时刻s')/60;hi=n(r,'结束时刻s')/60;key='直连' if r['保障方式']=='直连' else r['中继架次编号']
        ax.barh(tids.index(r['运输架次编号']),hi-lo,left=lo,height=.7,color=colors[key],linewidth=0)
    ax.set(yticks=range(len(tids)),yticklabels=tids,xlabel='任务开始后的时间 / min',title='各运输架次的通信保障方式（按导出区间）',xlim=(0,finish+2));ax.invert_yaxis()
    ax.legend(handles=[Patch(color=v,label=k) for k,v in colors.items()],ncol=4,loc='upper left',bbox_to_anchor=(0,1.065),fontsize=9)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);save(fig,'图2_通信保障时间线')

    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    seconds=summary['按运输架次累计的保障时长s'];keys=['直连']+[r['中继架次编号'] for r in relays]
    vals=[seconds.get(k,0)/60 for k in keys]
    bars=axs[0].bar(keys,vals,color=[colors[k] for k in keys]);axs[0].bar_label(bars,fmt='%.1f',padding=3)
    axs[0].set(ylabel='累计保障时长 / 运输架次·min',ylim=(0,max(vals)*1.2),title='按运输架次累计的保障时长')
    for r in relays:
        rid=r['中继架次编号'];rows=[x for x in comm if x['中继架次编号']==rid];events=defaultdict(int)
        for x in rows:
            if n(x,'结束时刻s')>n(x,'开始时刻s'):
                events[n(x,'开始时刻s')]+=1;events[n(x,'结束时刻s')]-=1
        tt=sorted(events);vv=np.cumsum([events[t] for t in tt]);axs[1].step([0]+[t/60 for t in tt]+[finish],[0]+list(vv)+[0],where='post',label=rid,color=colors[rid])
    axs[1].set(xlabel='时间 / min',ylabel='同时保障运输架次数',title='各中继的同时保障任务数');axs[1].legend(fontsize=8)
    from matplotlib.ticker import MaxNLocator
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axs:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,'图3_通信保障工作量')

    # 经纬度转为相对O01局部距离，用于位置示意；不代表地图投影或覆盖边界。
    origin=nodes['O01'];rad=np.pi/180;R=6371.0088
    def xy(lon,lat):return R*(lon-origin['lon'])*rad*np.cos(origin['lat']*rad),R*(lat-origin['lat'])*rad
    fig,ax=plt.subplots(figsize=(8.5,7.2),layout='constrained')
    for s,r in nodes.items():
        x,y=xy(r['lon'],r['lat']);ax.scatter(x,y,s=120 if s=='O01' else 35,marker='*' if s=='O01' else 'o',color='#252D35' if s=='O01' else '#89979F',zorder=3)
        ax.annotate(s,(x,y),xytext=(5,5),textcoords='offset points',fontsize=8)
    for i,r in enumerate(relays):
        x,y=xy(n(r,'悬停经度'),n(r,'悬停纬度'));ax.scatter(x,y,marker='^',s=120,color=colors[r['中继架次编号']],label=r['中继架次编号'],zorder=4)
        ax.annotate(r['中继架次编号']+'\n海拔 '+f"{n(r,'悬停海拔m'):.1f} m",(x,y),xytext=(8,-28 if i==1 else 14),textcoords='offset points',fontsize=8,color=colors[r['中继架次编号']])
    ax.set(xlabel='相对O01的东西向距离 / km',ylabel='相对O01的南北向距离 / km',title='服务区与中继驻守点位置示意（不表示覆盖范围）');ax.set_aspect('equal');ax.margins(.17);ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=8)
    save(fig,'图4_中继驻守位置')

    fig,axs=plt.subplots(1,2,figsize=(10.5,4.5),layout='constrained')
    vals=[summary['运输能耗kWh'],summary['中继能耗kWh']]
    bars=axs[0].bar(['运输无人机','中继无人机'],vals,color=['#4477AA','#D18B27']);axs[0].bar_label(bars,labels=[f'{v:.3f}\n({v/sum(vals)*100:.2f}%)' for v in vals],padding=4)
    axs[0].set(ylabel='能耗 / kWh',ylim=(0,max(vals)*1.25),title=f"总能耗 {sum(vals):.3f} kWh")
    bars=axs[1].bar([r['中继架次编号'] for r in relays],[n(r,'架次能耗kWh') for r in relays],color=[colors[r['中继架次编号']] for r in relays]);axs[1].bar_label(bars,fmt='%.3f',padding=4)
    axs[1].set(ylabel='能耗 / kWh',ylim=(0,max(n(r,'架次能耗kWh') for r in relays)*1.2),title='中继架次能耗分布')
    for ax in axs:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,'图5_运输中继能耗构成')

    fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained');names=[r['方案'] for r in comparison];idx=np.arange(len(names))
    times=[n(r,'联合完成时间s')/60 for r in comparison];bars=axs[0].barh(idx,times,color=['#6655AA','#779AAF','#779AAF']);axs[0].bar_label(bars,fmt='%.2f',padding=4)
    axs[0].set(yticks=idx,yticklabels=names,xlabel='联合完成时间 / min',xlim=(0,max(times)*1.17));axs[0].invert_yaxis()
    transport=[n(r,'运输能耗kWh') for r in comparison];relay=[n(r,'中继能耗kWh') for r in comparison]
    axs[1].barh(idx,transport,color='#4477AA',label='运输');axs[1].barh(idx,relay,left=transport,color='#D18B27',label='中继')
    for i,r in enumerate(comparison):axs[1].text(transport[i]+relay[i]+.6,i,f"{n(r,'总能耗kWh'):.2f}",va='center',fontsize=9)
    axs[1].set(yticks=idx,yticklabels=['①','②','③'],xlabel='总能耗 / kWh',xlim=(0,max(x+y for x,y in zip(transport,relay))*1.15));axs[1].invert_yaxis();axs[1].legend()
    axs[0].set_yticklabels([f'{i+1}. {name}' for i,name in enumerate(names)])
    for ax in axs:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('已发布Q3方案的联合完成时间与能耗比较');save(fig,'图6_协同方案比较')
    manifest={'repository':'https://github.com/jinding-droid/Drone-transportation','reference_commit':'9b407179971ab49327fc2d23b70ac620ae1c0856',
        'inputs':{f:hashlib.sha256((a.results/f).read_bytes()).hexdigest() for f in FILES},'nodes_sha256':hashlib.sha256(nodepath.read_bytes()).hexdigest(),
        'node_source':nodeinfo['source'],'node_source_sha256':nodeinfo['source_sha256'],'summary':summary,'font':font,'dpi':a.dpi,
        'limitations':'导出通信区间一致性核验不等于连续时间链路认证；未重算地形、链路损耗、能耗和电池/组件充电。位置图为局部近似坐标，海拔不是离地高度。'}
    (a.out/'数据核对与来源.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('输出目录',a.out.resolve())

if __name__=='__main__':main()
