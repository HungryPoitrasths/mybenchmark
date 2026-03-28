#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from string import Template

import numpy as np


PAGE = Template("""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>$title</title><style>
html,body{height:100%;margin:0;font:14px/1.4 "Segoe UI",Arial,sans-serif;background:#e8edf2;color:#102033}*{box-sizing:border-box}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:16px;height:100vh;padding:16px}
.card{background:rgba(255,255,255,.95);border:1px solid rgba(15,23,42,.08);border-radius:16px;box-shadow:0 16px 40px rgba(16,32,51,.12);overflow:hidden}
.view{display:grid;grid-template-rows:auto 1fr}.bar{display:flex;justify-content:space-between;gap:8px;align-items:center;padding:12px 14px;border-bottom:1px solid rgba(15,23,42,.08)}
.ttl h1{margin:0;font-size:20px}.ttl p{margin:3px 0 0;color:#5a6b7d;font-size:12px}.btns{display:flex;flex-wrap:wrap;gap:6px}
button{border:1px solid rgba(11,107,203,.2);background:#fff;border-radius:999px;padding:7px 11px;cursor:pointer}button.on{background:rgba(11,107,203,.12);color:#0b6bcb}
.stage{position:relative;background:#f7fafc}.vp{width:100%;height:100%}.labels{position:absolute;inset:0;pointer-events:none;overflow:hidden}
.lb{position:absolute;transform:translate(-50%,-50%);background:rgba(16,32,51,.8);color:#fff;padding:2px 7px;border-radius:999px;font-size:12px;white-space:nowrap}.lb.sel{background:#0b6bcb}
.side{display:grid;grid-template-rows:auto auto auto 1fr auto}.sec{padding:12px 14px;border-bottom:1px solid rgba(15,23,42,.08)}.sec h2{margin:0 0 4px;font-size:18px}.muted{color:#5a6b7d;font-size:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.box{background:#f8fafc;border:1px solid rgba(15,23,42,.06);border-radius:12px;padding:9px 10px}.k{color:#5a6b7d;font-size:12px}.v{font-weight:600;word-break:break-word}
input{width:100%;padding:10px 12px;border-radius:12px;border:1px solid rgba(15,23,42,.1)}.list{overflow:auto;padding:10px 10px 14px}.item{width:100%;text-align:left;border-radius:12px;border:1px solid rgba(15,23,42,.08);background:#fff;padding:10px;margin-bottom:8px}
.item.sel{background:rgba(11,107,203,.1);border-color:rgba(11,107,203,.3)}.row{display:flex;justify-content:space-between;gap:8px;align-items:center}.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
@media(max-width:1100px){.wrap{grid-template-columns:1fr;grid-template-rows:minmax(460px,62vh) auto;height:auto;min-height:100vh}}
</style></head><body><div class="wrap">
<section class="card view"><div class="bar"><div class="ttl"><h1>$scene_id</h1><p>$object_count objects · sampled mesh + 3D bbox viewer</p></div><div class="btns">
<button id="bReset">Reset</button><button id="bMesh" class="on">Mesh</button><button id="bBox" class="on">BBox</button><button id="bLabel" class="on">Label</button><button id="bFit">Fit Selected</button>
</div></div><div class="stage"><div id="vp" class="vp"></div><div id="labels" class="labels"></div></div></section>
<aside class="card side"><div class="sec"><h2>Object List</h2><div class="muted">Click an object to highlight sampled mesh points, label, and bbox.</div></div>
<div class="sec"><div class="grid">
<div class="box"><div class="k">Scene</div><div class="v">$scene_id</div></div><div class="box"><div class="k">Objects</div><div class="v">$object_count</div></div>
<div class="box"><div class="k">Mesh Vertices</div><div class="v" id="sVerts">-</div></div><div class="box"><div class="k">Sampled Points</div><div class="v" id="sPts">-</div></div>
<div class="box"><div class="k">Scene Min</div><div class="v" id="sMin">-</div></div><div class="box"><div class="k">Scene Max</div><div class="v" id="sMax">-</div></div>
</div></div>
<div class="sec"><div class="grid">
<div class="box"><div class="k">Label</div><div class="v" id="dLabel">-</div></div><div class="box"><div class="k">ID</div><div class="v" id="dId">-</div></div>
<div class="box"><div class="k">Center</div><div class="v" id="dCenter">-</div></div><div class="box"><div class="k">Dims</div><div class="v" id="dDims">-</div></div>
<div class="box"><div class="k">BBox Min</div><div class="v" id="dMin">-</div></div><div class="box"><div class="k">BBox Max</div><div class="v" id="dMax">-</div></div>
<div class="box"><div class="k">Vertices</div><div class="v" id="dVerts">-</div></div><div class="box"><div class="k">Samples</div><div class="v" id="dPts">-</div></div>
</div></div>
<div class="sec"><input id="q" type="search" placeholder="Filter by label or id"></div><div id="list" class="list"></div>
<div class="sec muted">Offline HTML. Mesh preview is sampled from ScanNet PLY vertices. Left drag: orbit. Right drag: pan. Wheel: zoom.</div></aside></div>
<script>
const D=$scene_json,O=Array.isArray(D.objects)?D.objects.slice():[],M=D.mesh_preview||{points:[],total_vertices:0,sample_count:0},P=D.object_point_samples||{};
let sel=null,showMesh=true,showBox=true,showLabel=true,cv,ctx,raf=null;const ll=document.getElementById("labels"),list=document.getElementById("list");
const cam={yaw:-.86,pitch:.48,r:10,t:[0,0,0],fov:Math.PI/3,near:.05},ptr={m:null,x:0,y:0};const ents=[];
const el=id=>document.getElementById(id),fmt=a=>a.map(v=>Number(v).toFixed(2)).join(", "),num=v=>Number(v||0).toLocaleString("en-US");
init();
function init(){const vp=el("vp");cv=document.createElement("canvas");cv.style.cssText="width:100%;height:100%;display:block";vp.appendChild(cv);ctx=cv.getContext("2d");window.addEventListener("resize",resize);resize();build();bind();sum();fitScene();if(ents.length)pick(ents[0],false);draw()}
function build(){const b=bounds(),sz=sub(b.max,b.min),span=Math.max(sz[0],sz[1],sz[2],1);cam.t=mid(b.min,b.max);cam.r=span*1.65;O.slice().sort((a,b)=>String(a.label).localeCompare(String(b.label))||Number(a.id)-Number(b.id)).forEach((o,i)=>{const e={o,c:col(i),p:Array.isArray(P[String(o.id)])?P[String(o.id)]:[]};e.l=document.createElement("div");e.l.className="lb";e.l.textContent=`${o.label} #${o.id}`;ll.appendChild(e.l);ents.push(e)})}
function sum(){const b=bounds();el("sMin").textContent=fmt(b.min);el("sMax").textContent=fmt(b.max);el("sVerts").textContent=num(M.total_vertices);el("sPts").textContent=num(M.sample_count);list.innerHTML="";ents.forEach(e=>{const bt=document.createElement("button");bt.className="item";bt.dataset.id=String(e.o.id);bt.dataset.label=String(e.o.label).toLowerCase();bt.innerHTML=`<div class=row><div><span class=sw style="background:${e.c}"></span>${esc(e.o.label)}</div><div class=muted>#${esc(String(e.o.id))}</div></div><div class=muted>center: ${fmt(e.o.center)}<br>dims: ${fmt(e.o.dimensions)}<br>vertices: ${num(e.o.vertex_count||0)}</div>`;bt.onclick=()=>pick(e,true);e.b=bt;list.appendChild(bt)})}
function pick(e,fit){sel=e;ents.forEach(x=>{const on=x===e;x.l.classList.toggle("sel",on);x.b.classList.toggle("sel",on)});el("dLabel").textContent=e.o.label;el("dId").textContent=String(e.o.id);el("dCenter").textContent=fmt(e.o.center);el("dDims").textContent=fmt(e.o.dimensions);el("dMin").textContent=fmt(e.o.bbox_min);el("dMax").textContent=fmt(e.o.bbox_max);el("dVerts").textContent=num(e.o.vertex_count||0);el("dPts").textContent=num(e.p.length);updLabels();if(fit)fitObj(e);req()}
function bind(){el("q").oninput=()=>{const q=el("q").value.trim().toLowerCase();ents.forEach(e=>{e.b.style.display=!q||String(e.o.label).toLowerCase().includes(q)||String(e.o.id).includes(q)?"":"none"})};
el("bReset").onclick=fitScene;el("bFit").onclick=()=>sel&&fitObj(sel);
el("bMesh").onclick=()=>tog("bMesh","showMesh");el("bBox").onclick=()=>tog("bBox","showBox");el("bLabel").onclick=()=>{showLabel=!showLabel;el("bLabel").classList.toggle("on",showLabel);updLabels()};
cv.oncontextmenu=e=>e.preventDefault();cv.onmousedown=e=>{ptr.m=e.button===2?"pan":"orb";ptr.x=e.clientX;ptr.y=e.clientY};
window.onmouseup=()=>{ptr.m=null};window.onmousemove=e=>{if(!ptr.m)return;const dx=e.clientX-ptr.x,dy=e.clientY-ptr.y;ptr.x=e.clientX;ptr.y=e.clientY;if(ptr.m==="orb"){cam.yaw-=dx*.008;cam.pitch=clamp(cam.pitch+dy*.008,-1.45,1.45)}else{const b=basis(),s=cam.r*.0015;cam.t=add(cam.t,add(sc(b.r,-dx*s),sc(b.u,dy*s)))}req()};
cv.addEventListener("wheel",e=>{e.preventDefault();cam.r=clamp(cam.r*Math.exp(e.deltaY*.0012),.4,200);req()},{passive:false})}
function tog(id,key){if(key==="showMesh")showMesh=!showMesh;else showBox=!showBox;el(id).classList.toggle("on",key==="showMesh"?showMesh:showBox);req()}
function resize(){const vp=el("vp"),r=Math.min(window.devicePixelRatio||1,2);cv.width=Math.max(1,Math.round(vp.clientWidth*r));cv.height=Math.max(1,Math.round(vp.clientHeight*r));ctx.setTransform(r,0,0,r,0,0);req()}
function req(){if(raf!==null)return;raf=requestAnimationFrame(draw)}
function draw(){raf=null;const w=cv.clientWidth,h=cv.clientHeight;ctx.clearRect(0,0,w,h);ctx.fillStyle="#f7fafc";ctx.fillRect(0,0,w,h);const b=basis();grid(b,w,h);axes(b,w,h);if(showMesh)mesh(b,w,h);if(sel)objPts(sel,b,w,h);if(showBox){const ds=ents.map(e=>({e,d:depth(e.o,b)})).sort((a,b)=>b.d-a.d);ds.forEach(x=>{if(x.e!==sel)box(x.e,b,w,h)});if(sel)box(sel,b,w,h)}updLabels()}
function mesh(b,w,h){ctx.save();for(const p of M.points||[]){const q=proj(p,b,w,h);if(!q.ok)continue;const s=clamp(2.8-q.z*.055,1.15,2.6);ctx.fillStyle=p[3]||"#74879a";ctx.globalAlpha=.92;ctx.fillRect(q.x-s/2,q.y-s/2,s,s)}ctx.restore()}
function objPts(e,b,w,h){if(!e.p.length)return;ctx.save();for(const p of e.p){const q=proj(p,b,w,h);if(!q.ok)continue;const r=clamp(2.0-q.z*.035,.85,1.8);ctx.beginPath();ctx.fillStyle="rgba(11,107,203,.65)";ctx.arc(q.x,q.y,r,0,Math.PI*2);ctx.fill();ctx.lineWidth=.7;ctx.strokeStyle="rgba(255,255,255,.92)";ctx.stroke()}ctx.restore()}
function box(e,b,w,h){const cs=corners(e.o.bbox_min,e.o.bbox_max).map(p=>proj(p,b,w,h));if(cs.filter(x=>x.ok).length<2)return;ctx.save();if(sel===e){ctx.lineWidth=6;ctx.strokeStyle="rgba(255,255,255,.98)";ctx.globalAlpha=1;ED.forEach(([a,c])=>line2(cs[a],cs[c]));ctx.lineWidth=3;ctx.strokeStyle="#ff6b00";ED.forEach(([a,c])=>line2(cs[a],cs[c]));ctx.fillStyle="#ff6b00";cs.forEach(p=>{if(p.ok)ctx.fillRect(p.x-2.5,p.y-2.5,5,5)})}else{ctx.lineWidth=1.9;ctx.strokeStyle=e.c;ctx.globalAlpha=.94;ED.forEach(([a,c])=>line2(cs[a],cs[c]))}ctx.restore()}
function updLabels(){const b=basis(),w=cv.clientWidth,h=cv.clientHeight;ents.forEach(e=>{if(!showLabel){e.l.style.display="none";return}const q=proj(e.o.center,b,w,h);if(!q.ok){e.l.style.display="none";return}e.l.style.display="block";e.l.style.left=`${q.x}px`;e.l.style.top=`${q.y}px`})}
function fitScene(){fit(bounds(),true)} function fitObj(e){fit({min:e.o.bbox_min,max:e.o.bbox_max},false)}
function fit(b,wide){const c=mid(b.min,b.max),s=sub(b.max,b.min),r=Math.max(Math.hypot(s[0],s[1],s[2])*.72,wide?3:1);cam.t=c;cam.r=r*(wide?1.45:1.25);req()}
function bounds(){if(!O.length)return{min:[-1,-1,-1],max:[1,1,1]};const mn=O[0].bbox_min.map(Number),mx=O[0].bbox_max.map(Number);O.forEach(o=>{for(let i=0;i<3;i++){mn[i]=Math.min(mn[i],Number(o.bbox_min[i]));mx[i]=Math.max(mx[i],Number(o.bbox_max[i]))}});return{min:mn,max:mx}}
function basis(){const cp=Math.cos(cam.pitch),p=[cam.t[0]+cam.r*cp*Math.cos(cam.yaw),cam.t[1]+cam.r*cp*Math.sin(cam.yaw),cam.t[2]+cam.r*Math.sin(cam.pitch)],f=nrm(sub(cam.t,p));let r=nrm(cr(f,[0,0,1]));if(len(r)<1e-6)r=[1,0,0];const u=nrm(cr(r,f));return{p,f,r,u}}
function proj(pt,b,w,h){const d=sub(pt,b.p),xc=dot(d,b.r),yc=dot(d,b.u),zc=dot(d,b.f);if(zc<=cam.near)return{ok:false,x:0,y:0,z:zc};const foc=h/(2*Math.tan(cam.fov/2));return{ok:true,z:zc,x:w/2+xc*foc/zc,y:h/2-yc*foc/zc}}
function grid(b,w,h){const bd=bounds(),z=bd.min[2],sx=bd.max[0]-bd.min[0],sy=bd.max[1]-bd.min[1],sp=Math.max(sx,sy,1),pad=sp*.18,x0=bd.min[0]-pad,x1=bd.max[0]+pad,y0=bd.min[1]-pad,y1=bd.max[1]+pad;ctx.save();ctx.lineWidth=1;ctx.strokeStyle="rgba(182,195,209,.85)";for(let i=0;i<=14;i++){const t=i/14,x=x0+(x1-x0)*t,y=y0+(y1-y0)*t;line([x,y0,z],[x,y1,z],b,w,h);line([x0,y,z],[x1,y,z],b,w,h)}ctx.restore()}
function axes(b,w,h){const bd=bounds(),sp=Math.max(bd.max[0]-bd.min[0],bd.max[1]-bd.min[1],bd.max[2]-bd.min[2],1),o=[0,0,bd.min[2]];ctx.save();ctx.lineWidth=2;ctx.strokeStyle="#d14b45";line(o,[o[0]+sp*.18,o[1],o[2]],b,w,h);ctx.strokeStyle="#2f8f4e";line(o,[o[0],o[1]+sp*.18,o[2]],b,w,h);ctx.strokeStyle="#0b6bcb";line(o,[o[0],o[1],o[2]+sp*.18],b,w,h);ctx.restore()}
function corners(a,b){const[x0,y0,z0]=a,[x1,y1,z1]=b;return[[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]]}
function line(a,b,ba,w,h){line2(proj(a,ba,w,h),proj(b,ba,w,h))} function line2(a,b){if(!a.ok||!b.ok)return;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
function depth(o,b){return dot(sub(o.center||mid(o.bbox_min,o.bbox_max),b.p),b.f)} function mid(a,b){return a.map((v,i)=>(Number(v)+Number(b[i]))/2)} function sub(a,b){return a.map((v,i)=>Number(v)-Number(b[i]))}
function add(a,b){return[a[0]+b[0],a[1]+b[1],a[2]+b[2]]} function sc(a,s){return[a[0]*s,a[1]*s,a[2]*s]} function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
function cr(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]} function len(a){return Math.hypot(a[0],a[1],a[2])}
function nrm(a){const l=len(a);return l<1e-8?[0,0,0]:[a[0]/l,a[1]/l,a[2]/l]} function clamp(v,lo,hi){return Math.min(hi,Math.max(lo,v))}
function col(i){return`hsl(${(i*47)%360},68%,50%)`} function esc(s){return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#39;")}
const ED=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
</script></body></html>""")

TYPES={"char":("b",1),"int8":("b",1),"uchar":("B",1),"uint8":("B",1),"short":("h",2),"int16":("h",2),"ushort":("H",2),"uint16":("H",2),"int":("i",4),"int32":("i",4),"uint":("I",4),"uint32":("I",4),"float":("f",4),"float32":("f",4),"double":("d",8),"float64":("d",8)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a single-scene mesh + bbox viewer")
    p.add_argument("--scene_metadata", required=True)
    p.add_argument("--output", default="scene_bbox_viewer.html")
    p.add_argument("--title", default=None)
    p.add_argument("--mesh_path", default=None)
    p.add_argument("--segs_path", default=None)
    p.add_argument("--aggregation_path", default=None)
    p.add_argument("--meta_path", default=None)
    p.add_argument("--max_scene_points", type=int, default=30000)
    p.add_argument("--max_object_points", type=int, default=320)
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_html_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def infer_paths(
    meta: Path,
    scene_id: str,
    mesh: str | None,
    segs: str | None,
    agg: str | None,
    meta_txt: str | None,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    base = meta.parent

    def pick(arg: str | None, names: list[str]) -> Path | None:
        if arg:
            p = Path(arg)
            return p if p.exists() else None
        for name in names:
            p = base / name
            if p.exists():
                return p
        return None

    return (
        pick(mesh, [f"{scene_id}_vh_clean.ply", f"{scene_id}_vh_clean_2.ply"]),
        pick(segs, [f"{scene_id}_vh_clean.segs.json", f"{scene_id}_vh_clean_2.0.010000.segs.json", f"{scene_id}_vh_clean_2.segs.json"]),
        pick(agg, [f"{scene_id}_vh_clean.aggregation.json", f"{scene_id}.aggregation.json"]),
        pick(meta_txt, [f"{scene_id}.txt"]),
    )


def load_axis_alignment(meta_path: Path | None) -> np.ndarray:
    if meta_path is None or not meta_path.exists():
        return np.eye(4, dtype=np.float32)
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if "axisAlignment" in line:
            values = [float(x) for x in line.split("=")[1].strip().split()]
            return np.array(values, dtype=np.float32).reshape(4, 4)
    return np.eye(4, dtype=np.float32)


def parse_ply_header(path: Path) -> dict:
    with path.open("rb") as f:
        if f.readline().decode("ascii", errors="replace").strip() != "ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = None
        count = None
        props: list[tuple[str, str]] = []
        cur = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} has an incomplete PLY header")
            s = line.decode("ascii", errors="replace").strip()
            if s == "end_header":
                start = f.tell()
                break
            if not s or s.startswith("comment"):
                continue
            parts = s.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                cur = parts[1]
                if cur == "vertex":
                    count = int(parts[2])
            elif parts[0] == "property" and cur == "vertex":
                if parts[1] == "list":
                    raise ValueError(f"{path} uses unsupported list vertex properties")
                props.append((parts[2], parts[1]))
    if fmt != "binary_little_endian":
        raise ValueError(f"{path} uses unsupported PLY format: {fmt}")
    if count is None or not props:
        raise ValueError(f"{path} is missing vertex metadata")
    fields, stride = {}, 0
    for name, typ in props:
        if typ not in TYPES:
            raise ValueError(f"{path} uses unsupported PLY type: {typ}")
        c, sz = TYPES[typ]
        fields[name] = (stride, c)
        stride += sz
    for req in ("x", "y", "z"):
        if req not in fields:
            raise ValueError(f"{path} is missing vertex property {req}")
    return {"count": count, "fields": fields, "stride": stride, "start": start}


def load_seg_to_obj(agg_path: Path, object_ids: set[int]) -> dict[int, int]:
    data = load_json(agg_path)
    out: dict[int, int] = {}
    for group in data.get("segGroups", []):
        oid = int(group.get("objectId", -1))
        if oid not in object_ids:
            continue
        for seg in group.get("segments", []):
            out[int(seg)] = oid
    return out


def sample_indices(scene: dict, segs_path: Path, agg_path: Path, max_scene: int, max_obj: int) -> tuple[list[int], dict[int, list[int]], int]:
    segs = load_json(segs_path).get("segIndices", [])
    total = len(segs)
    scene_stride = max(1, math.ceil(total / max(max_scene, 1)))
    scene_idx = list(range(0, total, scene_stride))
    ids = {int(o["id"]) for o in scene.get("objects", [])}
    seg_to_obj = load_seg_to_obj(agg_path, ids)
    state = {}
    for obj in scene.get("objects", []):
        vc = int(obj.get("vertex_count") or 0)
        state[int(obj["id"])] = {"stride": max(1, math.ceil(max(vc, 1) / max(max_obj, 1))), "seen": 0, "idx": []}
    for i, seg in enumerate(segs):
        oid = seg_to_obj.get(int(seg))
        if oid is None:
            continue
        st = state[oid]
        if st["seen"] % st["stride"] == 0 and len(st["idx"]) < max_obj:
            st["idx"].append(i)
        st["seen"] += 1
    return scene_idx, {oid: st["idx"] for oid, st in state.items()}, total


def unpack(record: bytes, spec: tuple[int, str]) -> float | int:
    off, code = spec
    return struct.unpack_from(f"<{code}", record, off)[0]


def rgb_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def pt3(x: float, y: float, z: float) -> list[float]:
    return [round(float(x), 4), round(float(y), 4), round(float(z), 4)]


def read_points(
    mesh: Path,
    hdr: dict,
    scene_idx: list[int],
    obj_idx: dict[int, list[int]],
    axis_alignment: np.ndarray,
) -> tuple[list[list[float | str]], dict[str, list[list[float]]]]:
    need = set(scene_idx)
    for idxs in obj_idx.values():
        need.update(idxs)
    order = sorted(need)
    flds, stride, start = hdr["fields"], int(hdr["stride"]), int(hdr["start"])
    sx, sy, sz = flds["x"], flds["y"], flds["z"]
    sr, sg, sb = flds.get("red"), flds.get("green"), flds.get("blue")
    R = axis_alignment[:3, :3].astype(np.float32)
    t = axis_alignment[:3, 3].astype(np.float32)
    scene_set = set(scene_idx)
    obj_sets = {oid: set(v) for oid, v in obj_idx.items()}
    scene_map: dict[int, list[float | str]] = {}
    obj_pts = {str(oid): [] for oid in obj_idx}
    with mesh.open("rb") as f:
        for i in order:
            f.seek(start + i * stride)
            rec = f.read(stride)
            if len(rec) != stride:
                raise ValueError(f"{mesh} ended while reading vertex {i}")
            pos = np.array(
                [float(unpack(rec, sx)), float(unpack(rec, sy)), float(unpack(rec, sz))],
                dtype=np.float32,
            )
            x, y, z = (pos @ R.T + t).tolist()
            if i in scene_set:
                if sr and sg and sb:
                    c = rgb_hex(int(unpack(rec, sr)), int(unpack(rec, sg)), int(unpack(rec, sb)))
                else:
                    c = "#8ca0b3"
                scene_map[i] = pt3(x, y, z) + [c]
            for oid, idxs in obj_sets.items():
                if i in idxs:
                    obj_pts[str(oid)].append(pt3(x, y, z))
    return [scene_map[i] for i in scene_idx if i in scene_map], obj_pts


def build_preview(
    scene: dict,
    mesh: Path,
    segs: Path,
    agg: Path,
    meta_path: Path | None,
    max_scene: int,
    max_obj: int,
) -> tuple[dict, dict[str, list[list[float]]]]:
    hdr = parse_ply_header(mesh)
    scene_idx, obj_idx, seg_count = sample_indices(scene, segs, agg, max_scene, max_obj)
    scene_pts, obj_pts = read_points(
        mesh,
        hdr,
        scene_idx,
        obj_idx,
        load_axis_alignment(meta_path),
    )
    if seg_count != int(hdr["count"]):
        raise ValueError(f"Vertex count mismatch between {mesh.name} and {segs.name}")
    return {"total_vertices": int(hdr["count"]), "sample_count": len(scene_pts), "points": scene_pts}, obj_pts


def main() -> None:
    a = parse_args()
    meta = Path(a.scene_metadata)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene = load_json(meta)
    scene_id = str(scene.get("scene_id") or meta.stem)
    mesh, segs, agg, meta_path = infer_paths(
        meta,
        scene_id,
        a.mesh_path,
        a.segs_path,
        a.aggregation_path,
        a.meta_path,
    )
    preview, obj_pts = {"total_vertices": 0, "sample_count": 0, "points": []}, {}
    if mesh and segs and agg:
        preview, obj_pts = build_preview(
            scene,
            mesh,
            segs,
            agg,
            meta_path,
            a.max_scene_points,
            a.max_object_points,
        )
        print(f"embedded mesh preview from {mesh.name}: {preview['sample_count']} sampled scene points")
    else:
        print("raw mesh files not found; generating bbox-only viewer data")
    scene = dict(scene)
    scene["mesh_preview"] = preview
    scene["object_point_samples"] = obj_pts
    html = PAGE.safe_substitute(title=a.title or f"{scene_id} mesh + bbox viewer", scene_id=scene_id, object_count=len(scene.get("objects") or []), scene_json=dump_html_json(scene))
    out.write_text(html, encoding="utf-8")
    print(f"wrote viewer to {out}")


if __name__ == "__main__":
    main()
