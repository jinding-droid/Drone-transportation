#!/usr/bin/env python3
"""问题三：DEM 链路仿真、通信中继与运输联合资源调度。"""
import argparse
import itertools
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from openpyxl import Workbook,load_workbook

import q1_solution as q1
import q2_solution as q2

def xyz(node,alt=None):
    return (node['lon'],node['lat'],node['elev'] if alt is None else alt)

def terrain_link(a,b,dem,threshold,loss_obs=10,step_pixels=.35):
    """链路双向门限；栅格线性近邻采样判定遮挡，FSPL 以 2400 MHz 计算。"""
    ax,ay,az=a;bx,by,bz=b
    horizontal=2*6371008.8*math.asin(math.sqrt(
        math.sin(math.radians(by-ay)/2)**2+
        math.cos(math.radians(ay))*math.cos(math.radians(by))*math.sin(math.radians(bx-ax)/2)**2))
    distance=math.hypot(horizontal,bz-az)
    if distance<1e-6:return True,0.,False
    lons=dem['longitude'].ravel();lats=dem['latitude'].ravel();z=dem['dem']
    n=max(2,math.ceil(max(abs(bx-ax)/abs(lons[1]-lons[0]),
                         abs(by-ay)/abs(lats[1]-lats[0]))/step_pixels))
    fraction=np.arange(1,n,dtype=float)/n
    xs=ax+(bx-ax)*fraction;ys=ay+(by-ay)*fraction
    ix=np.clip(np.rint((xs-lons[0])/(lons[1]-lons[0])).astype(int),0,len(lons)-1)
    iy=np.clip(np.rint((lats[0]-ys)/(lats[0]-lats[1])).astype(int),0,len(lats)-1)
    ground=z[iy,ix]
    if np.any(ground==float(dem['nodata'].ravel()[0])):return False,math.inf,True
    blocked=bool(np.any(ground>=az+(bz-az)*fraction))
    loss=32.44+20*math.log10(2400)+20*math.log10(distance/1000)+(loss_obs if blocked else 0)
    return loss<=threshold+1e-9,loss,blocked

def communication_params(root):
    s=load_workbook(q1.find(root,'通信链路参数.xlsx'),data_only=True).active
    param={}
    for row in list(s.values)[2:]:
        if row[0] and row[3]:param[(row[0],row[3])]=row[4]
    # 有效接收门限 -98 + 8 = -90 dBm；双向取更小允许损耗。
    rx=float(param[('接收参数','Psens')])+float(param[('接收参数','M')])
    sysloss=float(param[('传播参数','Lsys')])
    def budget(tx1,gain1,tx2,gain2):
        return min(tx1+gain1+gain2-sysloss-rx,tx2+gain2+gain1-sysloss-rx)
    gateway=(param[('固定网关 G01','Pt')],param[('固定网关 G01','G')])
    transport=(param[('运输无人机','Pt')],param[('运输无人机','G')])
    access=(param[('中继接入端','Pt')],param[('中继接入端','G')])
    backhaul=(param[('中继回传端','Pt')],param[('中继回传端','G')])
    return dict(direct=budget(*transport,*gateway),access=budget(*transport,*access),
                backhaul=budget(*backhaul,*gateway),gateway_alt=param[('固定网关 G01','hG')],
                obstruction=float(param[('传播参数','Lobs')]))

def samples(trip,boxes,models,nodes,dem,cache,dt=8):
    model=models[trip['model']]
    route=trip['route'];ids=trip['ids']
    t=trip['start']+model['prep']+model['loading']*len(ids)
    positions=[]
    last='O01'
    for nxt in route+['O01']:
        distance,up,down,alt=q2.leg(nodes,dem,last,nxt,cache)
        phases=[('爬升',up/model['climb']),('巡航',distance/model['speed']),('下降',down/model['descent'])]
        z0=nodes[last]['elev']+(0 if last=='O01' else 30)
        z1=nodes[nxt]['elev']+(0 if nxt=='O01' else 30)
        for label,duration in phases:
            count=max(1,int(math.ceil(duration/dt)))
            for k in range(count+1):
                frac=k/count
                now=t+duration*frac
                if label=='爬升':pos=(nodes[last]['lon'],nodes[last]['lat'],z0+(alt-z0)*frac)
                elif label=='巡航':pos=(nodes[last]['lon']+(nodes[nxt]['lon']-nodes[last]['lon'])*frac,
                                        nodes[last]['lat']+(nodes[nxt]['lat']-nodes[last]['lat'])*frac,alt)
                else:pos=(nodes[nxt]['lon'],nodes[nxt]['lat'],alt+(z1-alt)*frac)
                positions.append((now,label,pos))
            t+=duration
        if nxt!='O01':
            count_boxes=sum(boxes[i]['site']==nxt for i in ids)
            duration=model['handoff']+model['perbox']*count_boxes
            count=max(1,int(math.ceil(duration/dt)))
            for k in range(count+1):positions.append((t+k*duration/count,'交接',xyz(nodes[nxt],z1)))
            t+=duration
        last=nxt
    if abs(t-trip['end'])>.02:raise AssertionError((trip,t))
    return positions

def from_q2(path,boxes,models,nodes,dem,cache):
    wb=load_workbook(path,data_only=True)
    assigned=defaultdict(list)
    for row in list(wb['Q2_逐箱交付'].values)[1:]:assigned[row[1]].append(row[0])
    trips=[]
    for row in list(wb['Q2_运输架次'].values)[1:]:
        code,drone,kind,battery,start,route,end,energy=row
        t=dict(code=code,drone=drone,model=kind,battery=battery,start=start,
               route=route.split('→'),end=end,energy=energy,ids=assigned[code])
        t['positions']=samples(t,boxes,models,nodes,dem,cache)
        trips.append(t)
    return trips

def relay_data(root):
    s=load_workbook(q1.find(root,'中继无人机数据.xlsx'),data_only=True).active
    row=list(s.values)[2]
    out=dict(mass=row[4],speed=row[5],power=row[6],battery=row[7],reserve=row[8]/100,
             prep=row[9],link_time=row[10],turn=row[11],climb=row[12],descent=row[13],
             eta=row[14],hover=row[16],communication=row[17],max_height=row[18])
    inventory=list(s.values)[11]
    out['count']=int(inventory[1]);out['charge_full']=inventory[2]
    return out

def candidate_points(nodes,dem,relay,max_samples=500):
    # 候选悬停点在 DEM 内；经纬度为栅格中心，离地高度严格 <= 300 m。
    lons=dem['longitude'].ravel();lats=dem['latitude'].ravel();z=dem['dem']
    spots=[]
    # 调度中心及各服务区附近，在 0/150/300 m 离地高度采样。
    mid=[nodes['O01']]+[nodes[s] for s in sorted(nodes) if s!='O01']
    mid+= [dict(lon=(nodes['O01']['lon']+v['lon'])/2,lat=(nodes['O01']['lat']+v['lat'])/2)
           for s,v in nodes.items() if s!='O01']
    for v in mid:
        col=int(np.rint((v['lon']-lons[0])/(lons[1]-lons[0])))
        row=int(np.rint((lats[0]-v['lat'])/(lats[0]-lats[1])))
        for dr,dc in [(0,0),(-25,0),(25,0),(0,-25),(0,25)]:
            y=row+dr;x=col+dc
            if not (0<=y<len(lats) and 0<=x<len(lons)):continue
            if z[y,x]==float(dem['nodata'].ravel()[0]):continue
            for h in [150,300]:spots.append((float(lons[x]),float(lats[y]),float(z[y,x]+h),h))
    return list(dict.fromkeys(spots))[:max_samples]

SPOTS={
    'E':(109.28305555555556,23.019444444444446,717.970947265625),
    'W':(109.19972222222222,23.014722222222222,703.1117553710938),
    'N':(109.23805555555556,23.070833333333333,544.2846069335938),
}


def relay_flight(spot,relay,nodes,dem,cache):
    location=dict(nodes)
    lon,lat,alt=spot
    location['H']=dict(lon=lon,lat=lat,elev=alt-30)
    # 不同悬停点不能复用同名 H 的航段缓存。
    local_cache={}
    result=[]
    for origin,destination in [('O01','H'),('H','O01')]:
        d,up,down,cruise=q2.leg(location,dem,origin,destination,local_cache)
        seconds=d/relay['speed']+up/relay['climb']+down/relay['descent']
        energy=(relay['power']*d/relay['speed']/3600+
                relay['mass']*q1.G*up/(3.6e6*relay['eta']))
        result.append((seconds,energy,cruise,d,up,down))
    return result


def build_relay_mission(name,unit,component,start,service_end,relay,nodes,dem,cache):
    out,back=relay_flight(SPOTS[name],relay,nodes,dem,cache)
    arrive=start+relay['prep']+out[0]
    connected=arrive+relay['link_time']
    if service_end<connected:raise RuntimeError('中继服务结束早于建链')
    finish=service_end+back[0]
    energy=(out[1]+back[1]+relay['hover']*relay['link_time']/3600+
            (relay['hover']+relay['communication'])*(service_end-connected)/3600)
    if energy>relay['battery']*(1-relay['reserve'])+1e-6:
        raise RuntimeError(f'中继 {name} 超出返航安全余量：{energy:.3f} kWh')
    return dict(spot=name,unit=unit,component=component,start=start,arrive=arrive,
                connected=connected,service_end=service_end,finish=finish,energy=energy,
                soc=100*(1-energy/relay['battery']),out=out,back=back,
                battery_ready=finish+q2.charge_time(energy,relay['battery'],relay['charge_full']))


def change_transport(trips,boxes,models,nodes,dem,cache,requested):
    """保留 Q2 货箱组批/路线；调换 U08 两个任务并按资源就绪顺延。"""
    fleet_available=defaultdict(float);battery_available=defaultdict(float)
    original={t['code']:t for t in trips}
    start_ref={t['code']:t['start'] for t in trips}
    start_ref.update(requested)
    # Q2-020 原先在 Q2-015 之后；将两架次交换，提前 S003 的普通饮用水。
    start_ref['Q2-020']=original['Q2-015']['start']
    start_ref['Q2-015']=start_ref['Q2-020']+original['Q2-020']['end']-original['Q2-020']['start']
    battery_override={'Q2-020':'C-BAT-02','Q2-015':'C-BAT-03'}
    last_done={}
    planned=[]
    for code in sorted(original,key=lambda x:(start_ref[x],x)):
        old=original[code];kind=old['model'];drone=old['drone']
        battery=battery_override.get(code,old['battery'])
        starts=max(start_ref[code],fleet_available[drone],battery_available[battery])
        p=q2.profile(models[kind],old['route'],old['ids'],boxes,nodes,dem,cache)
        if p is None:raise AssertionError(code)
        t=dict(old,code=code,start=starts,end=starts+p['duration'],battery=battery,
               energy=p['energy'],deliveries={i:starts+v for i,v in p['deliveries'].items()},
               profile=p)
        fleet_available[drone]=t['end']
        battery_available[battery]=t['end']+q2.charge_time(p['energy'],models[kind]['battery'],
            {'A':1800,'B':2400,'C':3000}[kind])
        t['battery_ready']=battery_available[battery]
        t['positions']=samples(t,boxes,models,nodes,dem,cache,dt=1)
        planned.append(t)
    return sorted(planned,key=lambda t:(t['start'],t['code']))


def classify_outages(trips,nodes,dem,comms):
    gateway=xyz(nodes['O01'],nodes['O01']['elev']+comms['gateway_alt'])
    needs=defaultdict(list)
    masks=[]
    for trip in trips:
        for at,phase,pos in trip['positions']:
            direct=terrain_link(gateway,pos,dem,comms['direct'],comms['obstruction'])[0]
            if direct:continue
            mask=tuple(name for name,spot in SPOTS.items()
                       if terrain_link(spot,pos,dem,comms['access'],comms['obstruction'])[0])
            if not mask:raise RuntimeError(f"{trip['code']} {at:.2f}s 任一候选中继点均不可达")
            needs[trip['code']].append((at,phase,mask))
            masks.append((at,trip['code'],mask))
    return needs,masks


def schedule_joint(q2trips,models,nodes,dem,boxes,relay,comms,cache):
    # 第 1 次搜索运输延迟时先以粗样本估计需求窗口。
    fast=from_q2(Path('Q2_结果.xlsx'),boxes,models,nodes,dem,cache)
    initial,masks=classify_outages(fast,nodes,dem,comms)
    east_last=max(t for t,code,mask in masks if code in {'Q2-001','Q2-002','Q2-003','Q2-004','Q2-005','Q2-010','Q2-011','Q2-013'} and 'E' in mask)
    east_back=relay_flight(SPOTS['E'],relay,nodes,dem,cache)[1][0]
    north_out=relay_flight(SPOTS['N'],relay,nodes,dem,cache)[0][0]
    north_ready=east_last+20+east_back+relay['turn']+relay['prep']+north_out+relay['link_time']
    first_north={code:min(t for t,c,mask in masks if c==code and 'N' in mask)
                 for code in ['Q2-017','Q2-018']}
    target={
        'Q2-005':next(t['start'] for t in q2trips if t['code']=='Q2-005')+50,
        'Q2-017':next(t['start'] for t in q2trips if t['code']=='Q2-017')+
                  max(0,north_ready+45-first_north['Q2-017']),
        'Q2-018':next(t['start'] for t in q2trips if t['code']=='Q2-018')+
                  max(0,north_ready+45-first_north['Q2-018']),
    }
    trips=change_transport(q2trips,boxes,models,nodes,dem,cache,target)
    needs,masks=classify_outages(trips,nodes,dem,comms)
    # 同时运行的东/西中继为 R01/R02；R01 返回后周转到北侧。
    east_end=max(t for t,code,mask in masks if code in {'Q2-001','Q2-002','Q2-003','Q2-004','Q2-005','Q2-010','Q2-011','Q2-013'} and 'E' in mask)+20
    west_end=max(t for t,code,mask in masks if code in {'Q2-006','Q2-012','Q2-014','Q2-016','Q2-019','Q2-020'} and 'W' in mask)+20
    north_end=max(t for t,code,mask in masks if code in {'Q2-015','Q2-017','Q2-018','Q2-022'} and 'N' in mask)+20
    east=build_relay_mission('E','R01','R-BAT-01',0.,east_end,relay,nodes,dem,cache)
    west=build_relay_mission('W','R02','R-BAT-02',0.,west_end,relay,nodes,dem,cache)
    north_start=east['finish']+relay['turn']
    north=build_relay_mission('N','R01','R-BAT-03',north_start,north_end,relay,nodes,dem,cache)
    missions=[east,west,north]
    return trips,missions,needs


def verify_joint(trips,missions,needs,boxes,models,nodes,dem,comms):
    # 每个带实际时标的样本都必须同时具备直连或同一架中继的两段双向可用链路。
    for mission in missions:
        assert mission['finish']>=mission['service_end']>=mission['connected']
        assert mission['soc']>=20-1e-6
        assert mission['spot'] in SPOTS
        longitude,latitude,alt=SPOTS[mission['spot']]
        lat_grid=dem['latitude'].ravel();lon_grid=dem['longitude'].ravel()
        rr=int(round((lat_grid[0]-latitude)/abs(lat_grid[1]-lat_grid[0])))
        cc=int(round((longitude-lon_grid[0])/abs(lon_grid[1]-lon_grid[0])))
        assert 0<=rr<len(lat_grid) and 0<=cc<len(lon_grid)
        assert alt-dem['dem'][rr,cc]<=300+1e-3
        gateway=xyz(nodes['O01'],nodes['O01']['elev']+comms['gateway_alt'])
        assert terrain_link(gateway,SPOTS[mission['spot']],dem,comms['backhaul'],comms['obstruction'])[0]
    relay_sorted=sorted((m for m in missions if m['unit']=='R01'),key=lambda x:x['start'])
    assert relay_sorted[0]['finish']+300<=relay_sorted[1]['start']+1e-6
    assert len({m['component'] for m in missions})==3
    chosen={}
    outages=0
    for trip in trips:
        for at,phase,mask in needs[trip['code']]:
            live=[m for m in missions if m['spot'] in mask and m['connected']-1e-6<=at<=m['service_end']+1e-6]
            if not live:raise RuntimeError(f"{trip['code']} 在 {at:.2f}s {phase} 通信中断，可见中继 {mask}")
            chosen[(trip['code'],at)]=live[0]['spot'];outages+=1
    qtrips=[dict(ids=t['ids'],route=t['route'],model=t['model'],drone=t['drone'],
                 battery=t['battery'],start=t['start'],end=t['end'],battery_ready=t['battery_ready'],
                 profile=t['profile'],deliveries=t['deliveries']) for t in trips]
    fleet=defaultdict(list)
    for t in trips:
        if t['drone'] not in fleet[t['model']]:fleet[t['model']].append(t['drone'])
    batteries={k:[(f'{k}-BAT-{i:02d}',{'A':1800,'B':2400,'C':3000}[k]) for i in range(1,{'A':6,'B':4,'C':4}[k]+1)] for k in models}
    q2.validate(qtrips,boxes,models,fleet,batteries)
    return chosen,outages


def make_result(path,trips,missions,boxes,models,nodes,dem,comms,chosen):
    wb=Workbook();transport=wb.active;transport.title='Q2_运输架次'
    transport.append(['架次编号','无人机编号','机型编号','电池编号','开始时刻（s）','访问服务区顺序','返回O01时刻（s）','架次能耗（kWh）'])
    individual=wb.create_sheet('Q2_逐箱交付');individual.append(['货箱编号','架次编号','服务区编号','交付完成时刻（s）'])
    relay_sheet=wb.create_sheet('Q3_中继架次')
    relay_sheet.append(['中继架次编号','中继无人机编号','能源组件编号','开始时刻（s）','悬停经度（°）','悬停纬度（°）','悬停海拔（m）','建链完成时刻（s）','服务结束时刻（s）','返回O01时刻（s）','架次能耗（kWh）'])
    comm_sheet=wb.create_sheet('Q3_通信保障');comm_sheet.append(['运输架次编号','通信阶段','开始时刻（s）','结束时刻（s）','保障方式','中继架次编号'])
    chk=wb.create_sheet('联合资源校验');chk.append(['类别','编号','资源','起始时刻s','结束时刻s','其他'])
    due=wb.create_sheet('时限检查');due.append(['货箱编号','服务区','实际送达s','期望时间s','硬截止s','是否满足期望时间','是否满足硬截止'])
    for t in trips:
        transport.append([t['code'],t['drone'],t['model'],t['battery'],t['start'],'→'.join(t['route']),t['end'],t['energy']])
        chk.append(['运输',t['code'],t['drone']+' / '+t['battery'],t['start'],t['end'],f"电池充满 {t['battery_ready']:.2f}s；SOC {t['profile']['soc']:.2f}%"])
        for i,at in sorted(t['deliveries'].items()):
            b=boxes[i];hd=q2.hard_deadline(b)
            individual.append([i,t['code'],b['site'],at])
            due.append([i,b['site'],at,b['desired'],None if math.isinf(hd) else hd,
                        '是' if at<=b['desired']+1e-6 else '否','是' if at<=hd+1e-6 else '否'])
    relay_codes={m['spot']:f'Q3-R-{idx:02d}' for idx,m in enumerate(missions,1)}
    for m in missions:
        lon,lat,alt=SPOTS[m['spot']]
        relay_sheet.append([relay_codes[m['spot']],m['unit'],m['component'],m['start'],lon,lat,alt,
                            m['connected'],m['service_end'],m['finish'],m['energy']])
        chk.append(['中继',relay_codes[m['spot']],m['unit']+' / '+m['component'],m['start'],m['finish'],
                    f"电池充满 {m['battery_ready']:.2f}s；SOC {m['soc']:.2f}%"])
    # 连续采样点等状态合并成通信时间段；相邻采样状态不同时取中点边界。
    for t in trips:
        points=[]
        for at,phase,pos in t['positions']:
            spot=chosen.get((t['code'],at))
            state=('直连',None) if spot is None else ('中继',relay_codes[spot])
            points.append((at,phase,state))
        if not points:continue
        start=points[0][0];old=points[0][2];phase=points[0][1]
        for prev,nxt in zip(points,points[1:]):
            if nxt[2]!=old or nxt[1]!=phase:
                boundary=(prev[0]+nxt[0])/2 if nxt[0]>prev[0]+1e-7 else nxt[0]
                if boundary>start+1e-8:comm_sheet.append([t['code'],phase,start,boundary,*old])
                start=boundary;old=nxt[2];phase=nxt[1]
        if points[-1][0]>start+1e-8:comm_sheet.append([t['code'],phase,start,points[-1][0],*old])
    total=sum(t['energy'] for t in trips)+sum(m['energy'] for m in missions)
    info=wb.create_sheet('综合指标')
    for key,value in [('交付箱数',len(boxes)),('运输架次数',len(trips)),('中继架次数',len(missions)),
                      ('运输末架返航时刻s',max(t['end'] for t in trips)),
                      ('联合任务完成时刻s',max([t['end'] for t in trips]+[m['finish'] for m in missions])),
                      ('运输能耗kWh',sum(t['energy'] for t in trips)),
                      ('中继能耗kWh',sum(m['energy'] for m in missions)),
                      ('联合能耗kWh',total),
                      ('期望时间超时箱数',sum(t['deliveries'][i]>boxes[i]['desired']+1e-6 for t in trips for i in t['ids'])),
                      ('链路校验采样间隔s','各阶段不大于1秒；边界数值近似，见解题说明')]:info.append([key,value])
    for s in wb:s.freeze_panes='A2';s.auto_filter.ref=s.dimensions
    path.parent.mkdir(parents=True,exist_ok=True);wb.save(path)
    print('已输出',path,'运输',len(trips),'中继',len(missions),'联合能耗',round(total,4),'末返',round(max(t['end'] for t in trips),2))


def run(root,baseline,result):
    models,nodes,dem,fleet,batteries,boxes=q2.get_inputs(root)
    relay=relay_data(root);comms=communication_params(root);cache={}
    q2trips=from_q2(baseline,boxes,models,nodes,dem,cache)
    trips,missions,needs=schedule_joint(q2trips,models,nodes,dem,boxes,relay,comms,cache)
    chosen,outages=verify_joint(trips,missions,needs,boxes,models,nodes,dem,comms)
    print('断直连采样点',outages,'中继:',[(m['spot'],round(m['connected']),round(m['service_end']),round(m['finish']),round(m['energy'],3)) for m in missions])
    make_result(result,trips,missions,boxes,models,nodes,dem,comms,chosen)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data',type=Path,default=Path('work'))
    a.add_argument('--q2',type=Path,default=Path('Q2_结果.xlsx'))
    a.add_argument('--out',type=Path,default=Path('Q3_结果.xlsx'))
    args=a.parse_args()
    if args.data.suffix.lower()=='.zip':
        import tempfile,zipfile,shutil
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with zipfile.ZipFile(args.data) as z:
                for item in z.infolist():
                    if item.is_dir() or item.filename.endswith('\\'):continue
                    components=item.filename.replace('\\','/').split('/')
                    if '..' in components or not all(components):raise ValueError('不安全压缩包路径')
                    p=root.joinpath(*components);p.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(item) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst)
            run(root,args.q2,args.out)
    else:run(args.data,args.q2,args.out)


if __name__=='__main__':
    main()
