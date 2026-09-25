"""Continuous-time sufficient certificates over a piecewise-linear flight path.

Terrain is piecewise constant on the given raster. The moving line of sight
sweeps a 3-D triangle. Clipping that triangle against each intersected pixel
proves clearance over the ENTIRE interval, including between sample times.
False means 'not certified', not necessarily an outage.
"""
import math
import numpy as np
import common_physics as physics


def clip(poly,axis,bound,keep_above):
    if not poly:return []
    result=[];prev=poly[-1]
    prev_in=(prev[axis]>=bound) if keep_above else (prev[axis]<=bound)
    for cur in poly:
        inside=(cur[axis]>=bound) if keep_above else (cur[axis]<=bound)
        if inside!=prev_in:
            f=(bound-prev[axis])/(cur[axis]-prev[axis])
            result.append(tuple(a+(b-a)*f for a,b in zip(prev,cur)))
        if inside:result.append(cur)
        prev,prev_in=cur,inside
    return result


def triangle_clear(fixed,b0,b1,dem):
    triangle=[]
    for p in (fixed,b0,b1):
        x,y=physics.grid_xy(p,dem);triangle.append((x,y,p[2]))
    nr,nc=dem['dem'].shape
    if any(x<-.5 or x>nc-.5 or y<-.5 or y>nr-.5 for x,y,z in triangle):return False
    r0=max(0,math.floor(min(p[1] for p in triangle)+.5)-1)
    r1=min(nr-1,math.floor(max(p[1] for p in triangle)+.5)+1)
    nodata=float(dem['nodata'].ravel()[0])
    for row in range(r0,r1+1):
        strip=clip(clip(triangle,1,row-.5,True),1,row+.5,False)
        if not strip:continue
        c0=max(0,math.floor(min(p[0] for p in strip)+.5)-1)
        c1=min(nc-1,math.floor(max(p[0] for p in strip)+.5)+1)
        for col in range(c0,c1+1):
            ground=float(dem['dem'][row,col])
            if not math.isfinite(ground) or ground==nodata:return False
            if ground+1e-6<min(p[2] for p in strip):continue
            part=clip(clip(strip,0,col-.5,True),0,col+.5,False)
            if part and min(p[2] for p in part)<=ground+1e-6:return False
    return True


def movement_bound(a,b):
    """Upper bound for the length of the affine lon/lat/altitude trajectory."""
    dlat=math.radians(b[1]-a[1]);dlon=math.radians(b[0]-a[0])
    # cos <= 1 is globally safe (slightly loose at the scenario latitude).
    horizontal=6371008.8*math.hypot(dlat,dlon)
    return math.hypot(horizontal,b[2]-a[2])


def certify_link(fixed,b0,b1,dem,limit,frequency=2400,obstruction=10):
    d0=math.hypot(physics.horizontal_distance(fixed,b0),fixed[2]-b0[2])
    d1=math.hypot(physics.horizontal_distance(fixed,b1),fixed[2]-b1[2])
    # Every trajectory point is within half the arc length of an endpoint.
    upper=max(d0,d1)+movement_bound(b0,b1)/2
    base=physics.loss_db(upper,frequency)
    if base+obstruction<=limit-1e-7:
        return {'method':'distance_bound_with_obstruction','margin_db':limit-base-obstruction}
    if base>limit-1e-7:return None
    midpoint=tuple((a+b)/2 for a,b in zip(b0,b1))
    if not physics.terrain_link(fixed,midpoint,dem,limit,obstruction,frequency)[0]:return None
    if triangle_clear(fixed,b0,b1,dem):
        return {'method':'swept_triangle_clearance_and_distance_bound','margin_db':limit-base}
    return None


def trajectory(trip,models,boxes,nodes,dem,leg_function,cache):
    m=models[trip['model']];now=m['prep']+m['loading']*len(trip['ids']);last='O01';stages=[]
    for nxt in trip['route']+['O01']:
        d,up,down,alt=leg_function(nodes,dem,last,nxt,cache)
        a,b=nodes[last],nodes[nxt]
        za=a['elev']+(0 if last=='O01' else 30);zb=b['elev']+(0 if nxt=='O01' else 30)
        phases=[('爬升',up/m['climb'],(a['lon'],a['lat'],za),(a['lon'],a['lat'],alt)),
                ('巡航',d/m['speed'],(a['lon'],a['lat'],alt),(b['lon'],b['lat'],alt)),
                ('下降',down/m['descent'],(b['lon'],b['lat'],alt),(b['lon'],b['lat'],zb))]
        if nxt!='O01':
            dt=m['handoff']+m['perbox']*sum(boxes[i]['site']==nxt for i in trip['ids'])
            pos=(b['lon'],b['lat'],zb);phases.append(('交接',dt,pos,pos))
        for phase,duration,p0,p1 in phases:
            if duration>1e-10:stages.append((now,now+duration,phase,p0,p1))
            now+=duration
        last=nxt
    return stages


def mix(a,b,f):return tuple(x+(y-x)*f for x,y in zip(a,b))


def coverage(stages,gateway,relay,dem,comms,direct_cache=None):
    """Return whole-interval certificates. No unchecked midpoint extrapolation."""
    if direct_cache is None:direct_cache={}
    out=[]
    def part(lo,hi,phase,a,b,depth=0):
        key=(a,b)
        if key not in direct_cache:
            direct_cache[key]=certify_link(gateway,a,b,dem,comms['direct'],comms['frequency'],comms['obstruction'])
        cert=direct_cache[key]
        if cert:
            out.append(dict(start=lo,end=hi,phase=phase,mode='直连',**cert));return True
        if relay is not None:
            cert=certify_link(relay,a,b,dem,comms['access'],comms['frequency'],comms['obstruction'])
            if cert:
                out.append(dict(start=lo,end=hi,phase=phase,mode='中继',**cert));return True
        if hi-lo<=.005 or depth>=20:return False
        # An unavailable midpoint disproves any certificate for this interval.
        mid=mix(a,b,.5)
        if not physics.terrain_link(gateway,mid,dem,comms['direct'],comms['obstruction'],comms['frequency'])[0]:
            if relay is None or not physics.terrain_link(relay,mid,dem,comms['access'],comms['obstruction'],comms['frequency'])[0]:return False
        t=(lo+hi)/2
        return part(lo,t,phase,a,mid,depth+1) and part(t,hi,phase,mid,b,depth+1)
    for lo,hi,phase,a,b in stages:
        n=1 if a==b else max(1,math.ceil((hi-lo)/16))
        for j in range(n):
            if not part(lo+(hi-lo)*j/n,lo+(hi-lo)*(j+1)/n,phase,mix(a,b,j/n),mix(a,b,(j+1)/n)):
                return None
    return out
