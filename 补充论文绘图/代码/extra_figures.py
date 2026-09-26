# -*- coding: utf-8 -*-
"""新增8组论文图，依赖同目录core.py/q3.py/q4.py及包内数据。"""
from pathlib import Path
import argparse,csv,json,hashlib
from collections import defaultdict,Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib import font_manager
import core,q3,q4

ROOT=Path(__file__).resolve().parents[1]
C=['#4477AA','#CC6677','#228877']
def read(name):
    with (ROOT/'结果'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def step(intervals):
    events=defaultdict(int)
    for lo,hi in intervals:
        lo,hi=round(lo,3),round(hi,3)
        if hi>lo:events[lo]+=1;events[hi]-=1
    ts=sorted(events);return np.array([0]+ts)/60,np.array([0]+list(np.cumsum([events[t] for t in ts])))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dpi',type=int,default=600);p.add_argument('--grid',type=int,default=65);p.add_argument('--dt',type=float,default=1);p.add_argument('--out',type=Path,default=ROOT/'结果/新增论文图');a=p.parse_args()
    if a.grid<10 or a.dt<=0 or a.dpi<=0:p.error('grid>=10，dt和dpi须为正')
    fonts={f.name for f in font_manager.fontManager.ttflist};font=next((s for s in ['Microsoft YaHei','SimHei','Noto Sans CJK SC','SimSun'] if s in fonts),None)
    if not font:raise RuntimeError('请安装中文字体：思源黑体/Noto Sans CJK SC')
    plt.rcParams.update({'font.family':font,'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'path','legend.frameon':False})
    a.out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(a.out/f'{name}.{ext}',dpi=a.dpi,bbox_inches='tight',facecolor='white')
        plt.close(fig);print('完成',name,flush=True)
    nodes=core.load_nodes();acs=core.load_aircraft_types();boxes={b['id']:b for b in core.load_boxes()};dem,lat,lon=core.load_dem();terrain=core.Terrain(dem,lat,lon)
    report={'reference_commit':'9b407179971ab49327fc2d23b70ac620ae1c0856','grid':a.grid,'trajectory_sampling_s':a.dt}
    # Q1：复用原内核的密采样口径，展示真实地形而不修改求解。
    fig,axs=plt.subplots(3,1,figsize=(10,8),layout='constrained')
    for ax,s in zip(axs,['S001','S004','S008']):
        o,b=nodes['O01'],nodes[s];seg=core.build_segment(terrain,o,b)
        count=max(2,int(np.ceil(max(abs(b.lat-o.lat)/abs(terrain.dlat),abs(b.lon-o.lon)/abs(terrain.dlon))*8))+2)
        _,_,zz=terrain.points_along(o.lon,o.lat,b.lon,b.lat,samples=count);zz=np.where(zz>-32766,zz,np.nan);xx=np.linspace(0,seg.horizontal_m/1000,len(zz))
        ax.fill_between(xx,zz,color='#B9C8B0',alpha=.8);ax.plot(xx,zz,color='#66795E',lw=.8,label='沿线DEM')
        ax.plot([0,0,xx[-1],xx[-1]],[o.op_alt_m,seg.cruise_alt_m,seg.cruise_alt_m,b.op_alt_m],color='#4477AA',label='模型去程飞行高度')
        ax.plot(xx,zz+50,ls=':',color='#AA7766',lw=.8,label='地面高程+50 m参考线')
        ax.set(title=f'O01 → {s}：模型巡航海拔 {seg.cruise_alt_m:.1f} m',xlabel='水平航程 / km',ylabel='海拔 / m');ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.15)
    save(fig,'Q1新增1_地形与飞行高度剖面')
    batches=read('Q1_组批方案.csv');mass=[];vol=[]
    for r in batches:
        bb=[boxes[x] for x in r['货箱编号列表'].split(',')];m=sum(x['mass_kg'] for x in bb);v=sum(x['volume_m3'] for x in bb);ac=acs[r['机型编号']]
        if abs(m-float(r['总质量kg']))>.01 or abs(v-float(r['总体积m³']))>.0001:raise ValueError('Q1装载与附件不符')
        mass.append(m/ac.max_payload_kg*100);vol.append(v/ac.volume_m3*100)
    if max(mass+vol)>100.001:raise ValueError('Q1超过额定装载限制')
    fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained');x=np.arange(len(batches));ax.bar(x-.19,mass,.36,label='额定载质量利用率',color=C[0]);ax.bar(x+.19,vol,.36,label='装载体积利用率',color=C[2]);ax.axhline(100,color='#555555',ls='--')
    ax.set(xticks=x,xticklabels=[r['架次编号']+'\n'+r['机型编号'] for r in batches],ylabel='利用率 / %',ylim=(0,118),title='逐架次质量与体积约束利用率');ax.legend(ncol=2);ax.grid(axis='y',alpha=.15);save(fig,'Q1新增2_质量体积双约束')
    # 地形图只裁到研究任务点附近，颜色表示海拔。
    xmin=min(n.lon for n in nodes.values())-.015;xmax=max(n.lon for n in nodes.values())+.015;ymin=min(n.lat for n in nodes.values())-.015;ymax=max(n.lat for n in nodes.values())+.015
    ii=np.where((lat>=ymin)&(lat<=ymax))[0][::3];jj=np.where((lon>=xmin)&(lon<=xmax))[0][::3]
    def background(ax):
        im=ax.pcolormesh(lon[jj],lat[ii],np.ma.masked_less(dem[np.ix_(ii,jj)],-32766),cmap='terrain',shading='auto',rasterized=True,alpha=.65)
        ax.set(xlabel='经度 / °E',ylabel='纬度 / °N',xlim=(xmin,xmax),ylim=(ymin,ymax));ax.set_aspect(1/np.cos(np.radians(nodes['O01'].lat)));return im
    t2=read('Q2_运输架次.csv');fig,axs=plt.subplots(1,3,figsize=(14,5.6),layout='constrained')
    for ax,g,color in zip(axs,'ABC',C):
        im=background(ax);counts=Counter()
        for t in t2:
            if t['机型编号']==g:
                route=['O01']+t['访问服务区顺序'].split(',')+['O01'];counts.update(zip(route,route[1:]))
        for (u,v),cnt in counts.items():
            p0,p1=nodes[u],nodes[v];ax.annotate('',xy=(p1.lon,p1.lat),xytext=(p0.lon,p0.lat),arrowprops={'arrowstyle':'->','color':color,'lw':.5+.45*cnt,'alpha':.8})
        for s,nod in nodes.items():ax.scatter(nod.lon,nod.lat,s=12,color='#202A33');ax.annotate(s,(nod.lon,nod.lat),xytext=(3,3),textcoords='offset points',fontsize=6)
        ax.set_title(g+'型运输航线（线宽表示使用次数）')
    fig.colorbar(im,ax=axs,label='地面海拔 / m',shrink=.7);save(fig,'Q2新增1_DEM运输航线')
    bats=read('Q2_电池资源台账.csv');fleet,stock,_=core.load_fleet();fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    q2peaks={}
    for j,g in enumerate('ABC'):
        for i,(rows,start,end,cap,label) in enumerate([(t2,'开始时刻s','返回O01时刻s',sum(t==g for _,t in fleet),'无人机'),(bats,'占用开始s','充电完成s',stock[g],'电池（含充电）')]):
            tt,vv=step([(float(r[start]),float(r[end])) for r in rows if r['机型编号']==g]);ax=axs[i,j];ax.step(tt,vv,where='post',color=C[j]);ax.axhline(cap,color='#B53A3A',ls='--',label='库存上限');ax.set(title=f'{g}型{label}：峰值{max(vv)} / 库存{cap}',xlabel='时间 / min',ylabel='占用数量',ylim=(0,cap+1));ax.grid(alpha=.15);q2peaks[g+label]=int(max(vv))
            if max(vv)>cap:raise ValueError('Q2并发超过库存')
    save(fig,'Q2新增2_资源并发占用');report['Q2资源峰值']=q2peaks
    # Q3图1：固定AGL切片，叠加不同时间片实际可用中继，不假设全部中继同时在线。
    link=q3.LinkModel();relays=read('Q3_中继架次.csv');snapshots=[1800,3000,6000];agl=50
    gx=np.linspace(xmin,xmax,a.grid);gy=np.linspace(ymin,ymax,a.grid);ground=np.array([[terrain.elevation(x,y) for x in gx] for y in gy])
    direct=np.zeros((a.grid,a.grid),dtype=bool);access=[]
    for r in relays:
        rp=q3.Point3D(float(r['悬停经度']),float(r['悬停纬度']),float(r['悬停海拔m']));back=link.available(rp,link.gateway,'backhaul');arr=np.zeros_like(direct)
        for i,y in enumerate(gy):
            for j,x in enumerate(gx):
                if ground[i,j]>-32766:arr[i,j]=back and link.available(q3.Point3D(x,y,ground[i,j]+agl),rp,'access')
        access.append(arr)
    for i,y in enumerate(gy):
        for j,x in enumerate(gx):
            if ground[i,j]>-32766:direct[i,j]=link.available(q3.Point3D(x,y,ground[i,j]+agl),link.gateway,'direct')
    fig,axs=plt.subplots(1,3,figsize=(14,5.5),layout='constrained');categories=[]
    for ax,ts in zip(axs,snapshots):
        background(ax);covered=np.zeros_like(direct);active=[]
        for r,arr in zip(relays,access):
            if float(r['建链完成时刻s'])<=ts<float(r['服务结束时刻s']):
                covered|=arr;active.append(r['中继架次编号']);ax.scatter(float(r['悬停经度']),float(r['悬停纬度']),marker='^',s=75,color='#161A20')
        z=np.where(direct,1,np.where(covered,2,0)).astype(float);z[ground<=-32766]=np.nan;categories.append(z)
        ax.pcolormesh(gx,gy,z,cmap=ListedColormap(['#CC6677','#B9C7D3','#228877']),vmin=-.5,vmax=2.5,shading='auto',alpha=.65,rasterized=True)
        ax.scatter(link.gateway.lon,link.gateway.lat,marker='*',s=80,color='#111111');ax.set_title(f't={ts/60:.0f} min\n在线：'+(','.join(active) or '无'),fontsize=9)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(['#CC6677','#B9C7D3','#228877'],['该切片不可用','可直连','仅中继可用'])],loc='outside lower center',ncol=3)
    fig.suptitle('固定离地50 m网格通信可用性：地形遮挡模型采样，非全空域覆盖');save(fig,'Q3新增1_通信盲区与中继改善')
    np.savez_compressed(a.out/'通信切片采样.npz',longitude=gx,latitude=gy,times_s=snapshots,agl_m=agl,categories=np.array(categories))
    # 真实Q3轨迹上的裕量，只在对应中继实际服务窗口内绘制接入/回传的较小值。
    t3=read('Q3_运输架次.csv')
    for t in t3:t.update(start_s=float(t['开始时刻s']),return_s=float(t['返回O01时刻s']),route=tuple(t['访问服务区顺序'].split(',')),box_ids=tuple(t['货箱编号列表'].split(',')))
    phases=q3.transport_phases(t3);cc=read('Q3_通信保障.csv');duration=defaultdict(float)
    for r in cc:
        if r['保障方式']=='中继':duration[r['运输架次编号']]+=float(r['结束时刻s'])-float(r['开始时刻s'])
    chosen=sorted(duration,key=duration.get,reverse=True)[:2];fig,axs=plt.subplots(2,1,figsize=(12,7.5),layout='constrained');samples=[]
    for ax,tid in zip(axs,chosen):
        ps=[p for p in phases if p.trip_id==tid];times=np.unique(np.concatenate([np.linspace(p.start_s,p.end_s,max(2,int(np.ceil((p.end_s-p.start_s)/a.dt))+1)) for p in ps]));dm=[];rm=[[] for r in relays]
        idx=0
        for ts in times:
            while idx<len(ps)-1 and ts>ps[idx].end_s+1e-8:idx+=1
            pos=ps[idx].point(ts);d=link.threshold_db('direct')-link.path_loss_db(pos,link.gateway);dm.append(d);row=[tid,ts,d]
            for k,r in enumerate(relays):
                val=np.nan
                if float(r['建链完成时刻s'])<=ts<=float(r['服务结束时刻s']):
                    rp=q3.Point3D(float(r['悬停经度']),float(r['悬停纬度']),float(r['悬停海拔m']));val=min(link.threshold_db('access')-link.path_loss_db(pos,rp),link.threshold_db('backhaul')-link.path_loss_db(rp,link.gateway))
                rm[k].append(val);row.append('' if np.isnan(val) else val)
            samples.append(row)
        ax.plot(times/60,dm,color='#657580',lw=1,label='直连裕量')
        for k,r in enumerate(relays):ax.plot(times/60,rm[k],color=C[k],lw=1,label=r['中继架次编号']+'两跳最小裕量')
        ax.axhline(0,color='#A93333',ls='--',lw=1);ax.set_yscale('symlog',linthresh=5);ax.set_yticks([-10,-5,0,5,10,20,50,100]);ax.set_yticklabels(['−10','−5','0','5','10','20','50','100']);ax.set(title=tid+'：按中继依赖时长选取的典型架次',xlabel='绝对时间 / min',ylabel='链路裕量 / dB（对称对数轴，±5内线性）');ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.15)
    save(fig,'Q3新增2_典型架次链路裕量')
    with (a.out/'链路裕量采样.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['架次','时间s','直连裕量dB']+[r['中继架次编号']+'裕量dB' for r in relays]);w.writerows(samples)
    report['链路曲线典型架次']=chosen
    # Q4：重新构造含充电和周转的区间，核对发布的每组峰值。
    manifest=json.loads((ROOT/'结果/Q4_input_manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['input_sha256'].items():
        if hashlib.sha256((ROOT/'结果'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Q4冻结输入不一致 '+name)
    model=q4.Q4Model();groups=read('Q4_分区配置.csv');fields=['A型运输无人机数','B型运输无人机数','C型运输无人机数','A型电池组数','B型电池组数','C型电池组数','中继无人机数','中继能源组件数']
    allintervals={};peakrows=[]
    for k in [2,3]:
        rr=[r for r in groups if int(r['K'])==k];out=[]
        for r in rr:
            ss=set(r['服务区列表'].split(','));selected=[t for t in model.trips if set(t['sites'])<=ss];rids=set().union(*(model.trip_relays[t['id']] for t in selected));iv=[]
            for g in 'ABC':iv.append([(t['start'],t['end']) for t in selected if t['type']==g])
            for g in 'ABC':iv.append([(t['start'],t['battery_end']) for t in selected if t['type']==g])
            iv.append([(model.relays[r]['start'],model.relays[r]['drone_end']) for r in rids]);iv.append([(model.relays[r]['start'],model.relays[r]['component_end']) for r in rids])
            if [q4.peak(x) for x in iv]!=[int(r[f]) for f in fields]:raise ValueError('重算组内峰值不一致')
            out.append(iv)
        allintervals[k]=out
        for j,name in enumerate(model.resource_names):
            separate=sum(q4.peak(iv[j]) for iv in out);combined=q4.peak([x for iv in out for x in iv[j]]);peakrows.append([k,name,combined,separate,separate-combined])
    fig,axs=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
    for col,k in enumerate([2,3]):
        out=allintervals[k]
        for row,j in enumerate([3,6]):
            ax=axs[row,col];union=sorted({0.}|{round(t,3) for iv in out for lohi in iv[j] for t in lohi});values=[]
            for iv in out:values.append([sum(round(lo,3)<=t<round(hi,3) for lo,hi in iv[j]) for t in union])
            ax.stackplot(np.array(union)/60,*values,step='post',colors=C[:k],labels=[f'G{i+1}' for i in range(k)],alpha=.8)
            sep=sum(q4.peak(iv[j]) for iv in out);combined=max(np.sum(values,axis=0));ax.axhline(sep,color='#A93333',ls='--',label=f'各组峰值之和={sep}');ax.axhline(combined,color='#333333',ls=':',label=f'合计占用峰值={combined}')
            ax.set(title=f'{k}组：{model.resource_names[j]}',xlabel='时间 / min',ylabel='占用数量',ylim=(0,sep+2));ax.legend(ncol=2,fontsize=8)
    fig.suptitle('独立配置为何增加：各组峰值之和与合计占用峰值');save(fig,'Q4新增1_分组资源峰值机制')
    with (a.out/'分组峰值重算.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['K','资源','分组后合计占用峰值','各组峰值之和','独立配置增量']);w.writerows(peakrows)
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained');edges=set()
    for t in model.trips:
        for u,v in zip(t['sites'],t['sites'][1:]):edges.add(tuple(sorted([u,v])))
    components=[[s for i,s in enumerate(q4.SITES) if mask&(1<<i)] for mask in model.components];positions={}
    for idx,unit in enumerate(sorted(components,key=lambda x:(-len(x),x))):
        cx=(idx%4)*3.2;cy=-(idx//4)*2.3
        for j,s in enumerate(unit):
            angle=2*np.pi*j/len(unit);positions[s]=(cx+(.9*np.cos(angle) if len(unit)>1 else 0),cy+(.7*np.sin(angle) if len(unit)>1 else 0))
        ax.text(cx,cy+1.1,'同组单元：'+str(len(unit))+'站',ha='center',fontsize=8,color='#666666')
    for u,v in edges:ax.plot([positions[u][0],positions[v][0]],[positions[u][1],positions[v][1]],color='#4477AA',lw=2,zorder=1)
    for s,(x,y) in positions.items():ax.scatter(x,y,s=850,color='#D6E6EE',edgecolor='#4477AA',zorder=2);ax.text(x,y,s,ha='center',va='center',fontsize=9,zorder=3)
    ax.set_title('运输同架次关系形成的11个不可拆分单元（布局无地理含义）');ax.axis('off');ax.margins(.1);save(fig,'Q4新增2_不可拆分关系图')
    report['Q4峰值重算']=peakrows;report['通信切片']='固定AGL50m；30/50/100分钟实际在线中继；离散网格；不是连续全空域证明'
    report['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['代码','数据','结果'] for p in (ROOT/folder).rglob('*') if p.is_file() and a.out not in p.parents and '__pycache__' not in str(p) and p.suffix in ['.py','.csv','.xlsx','.mat','.json']}
    (a.out/'来源与核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('全部完成：',a.out,flush=True)

if __name__=='__main__':main()
