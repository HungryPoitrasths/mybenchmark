#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
.stage{position:relative;background:#f7fafc}.gl,.ov{position:absolute;inset:0;width:100%;height:100%;display:block}.labels{position:absolute;inset:0;pointer-events:none;overflow:hidden}
.lb{position:absolute;transform:translate(-50%,-50%);background:rgba(16,32,51,.84);color:#fff;padding:2px 7px;border-radius:999px;font-size:12px;white-space:nowrap}
.side{display:grid;grid-template-rows:auto auto auto 1fr auto}.sec{padding:12px 14px;border-bottom:1px solid rgba(15,23,42,.08)}.sec h2{margin:0 0 4px;font-size:18px}.muted{color:#5a6b7d;font-size:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.box{background:#f8fafc;border:1px solid rgba(15,23,42,.06);border-radius:12px;padding:9px 10px}.k{color:#5a6b7d;font-size:12px}.v{font-weight:600;word-break:break-word}
input{width:100%;padding:10px 12px;border-radius:12px;border:1px solid rgba(15,23,42,.1)}.list{overflow:auto;padding:10px 10px 14px}.item{width:100%;text-align:left;border-radius:12px;border:1px solid rgba(15,23,42,.08);background:#fff;padding:10px;margin-bottom:8px}
.item.sel{background:rgba(11,107,203,.1);border-color:rgba(11,107,203,.3)}.row{display:flex;justify-content:space-between;gap:8px;align-items:center}.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
@media(max-width:1100px){.wrap{grid-template-columns:1fr;grid-template-rows:minmax(460px,62vh) auto;height:auto;min-height:100vh}}
</style></head><body><div class="wrap">
<section class="card view"><div class="bar"><div class="ttl"><h1>$scene_id</h1><p>$object_count objects · WebGL triangle mesh + aligned bbox viewer</p></div><div class="btns">
<button id="bReset">Reset</button><button id="bMesh" class="on">Mesh</button><button id="bBBox" class="on">BBox</button><button id="bLabel" class="on">Label</button><button id="bFit">Fit Selected</button>
</div></div><div class="stage"><canvas id="gl" class="gl"></canvas><canvas id="ov" class="ov"></canvas><div id="labels" class="labels"></div></div></section>
<aside class="card side"><div class="sec"><h2>Object List</h2><div class="muted">Selected object shows aligned 3D bbox, label and sampled vertices on top of the triangle mesh.</div></div>
<div class="sec"><div class="grid">
<div class="box"><div class="k">Scene</div><div class="v">$scene_id</div></div><div class="box"><div class="k">Objects</div><div class="v">$object_count</div></div>
<div class="box"><div class="k">Mesh Vertices</div><div class="v" id="sVerts">-</div></div><div class="box"><div class="k">Mesh Faces</div><div class="v" id="sFaces">-</div></div>
<div class="box"><div class="k">Aligned With</div><div class="v">axisAlignment</div></div><div class="box"><div class="k">Scene Type</div><div class="v">$scene_type</div></div>
<div class="box"><div class="k">Scene Min</div><div class="v" id="sMin">-</div></div><div class="box"><div class="k">Scene Max</div><div class="v" id="sMax">-</div></div>
</div></div>
<div class="sec"><div class="grid">
<div class="box"><div class="k">Label</div><div class="v" id="dLabel">-</div></div><div class="box"><div class="k">ID</div><div class="v" id="dId">-</div></div>
<div class="box"><div class="k">Center</div><div class="v" id="dCenter">-</div></div><div class="box"><div class="k">Dims</div><div class="v" id="dDims">-</div></div>
<div class="box"><div class="k">BBox Min</div><div class="v" id="dMin">-</div></div><div class="box"><div class="k">BBox Max</div><div class="v" id="dMax">-</div></div>
<div class="box"><div class="k">Vertices</div><div class="v" id="dVerts">-</div></div><div class="box"><div class="k">Sampled Verts</div><div class="v" id="dPts">-</div></div>
</div></div>
<div class="sec"><input id="q" type="search" placeholder="Filter by label or id"></div><div id="list" class="list"></div>
<div class="sec muted">Left drag: orbit. Right drag: pan. Wheel: zoom. This page is fully offline and embeds a sampled triangle mesh from the ScanNet PLY.</div></aside></div>
<script>
const SCENE=$scene_json,MESH=$mesh_json,OBJ=Array.isArray(SCENE.objects)?SCENE.objects.slice():[];
const glCanvas=document.getElementById("gl"),ovCanvas=document.getElementById("ov"),labels=document.getElementById("labels"),list=document.getElementById("list");
const ctx2d=ovCanvas.getContext("2d"),gl=glCanvas.getContext("webgl",{antialias:true,alpha:false});
let showMesh=true,showBBox=true,showLabel=true,raf=null,selected=null;
const cam={yaw:-0.92,pitch:0.48,r:8,t:[0,0,0],fov:Math.PI/3,near:.05,far:120},ptr={m:null,x:0,y:0};
const el=id=>document.getElementById(id),fmt=a=>a.map(v=>Number(v).toFixed(2)).join(", "),num=v=>Number(v||0).toLocaleString("en-US");
const ents=[]; let renderState=null;
init();
function init(){if(!gl){alert("WebGL is unavailable in this browser.");return;} setupGL(); buildEntries(); bind(); resize(); fillSummary(); fitScene(); if(ents.length) pick(ents[0],false); draw();}
function setupGL(){
  const vs=`attribute vec3 aPos;attribute vec3 aNormal;attribute vec3 aColor;uniform mat4 uMvp;uniform mat3 uN;varying vec3 vColor;varying float vL;void main(){vec3 n=normalize(uN*aNormal);vec3 l=normalize(vec3(0.32,0.58,0.75));vL=0.35+0.65*max(dot(n,l),0.0);vColor=aColor;gl_Position=uMvp*vec4(aPos,1.0);}`;
  const fs=`precision mediump float;varying vec3 vColor;varying float vL;void main(){gl_FragColor=vec4(vColor*vL,1.0);}`;
  const prog=mkProgram(vs,fs); gl.useProgram(prog); gl.enable(gl.DEPTH_TEST); gl.clearColor(.965,.973,.988,1);
  const posBuf=gl.createBuffer(), norBuf=gl.createBuffer(), colBuf=gl.createBuffer(), idxBuf=gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER,posBuf); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(MESH.positions),gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER,norBuf); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(MESH.normals),gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER,colBuf); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(MESH.colors),gl.STATIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,idxBuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint32Array(MESH.indices),gl.STATIC_DRAW);
  extUint();
  bindAttrib(prog,"aPos",posBuf,3); bindAttrib(prog,"aNormal",norBuf,3); bindAttrib(prog,"aColor",colBuf,3);
  renderState={prog,posBuf,norBuf,colBuf,idxBuf,uMvp:gl.getUniformLocation(prog,"uMvp"),uN:gl.getUniformLocation(prog,"uN"),indexCount:MESH.indices.length};
}
function extUint(){return gl.getExtension("OES_element_index_uint")}
function mkProgram(vsSrc,fsSrc){const v=mkShader(gl.VERTEX_SHADER,vsSrc),f=mkShader(gl.FRAGMENT_SHADER,fsSrc),p=gl.createProgram(); gl.attachShader(p,v); gl.attachShader(p,f); gl.linkProgram(p); if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p}
function mkShader(type,src){const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s); if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s}
function bindAttrib(prog,name,buf,size){const loc=gl.getAttribLocation(prog,name); gl.bindBuffer(gl.ARRAY_BUFFER,buf); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,size,gl.FLOAT,false,0,0)}
function buildEntries(){
  const sorted=OBJ.slice().sort((a,b)=>String(a.label).localeCompare(String(b.label))||Number(a.id)-Number(b.id));
  sorted.forEach((o,i)=>{const e={o,c:col(i),p:(SCENE.object_point_samples||{})[String(o.id)]||[]}; const d=document.createElement("div"); d.className="lb"; d.textContent=`${o.label} #${o.id}`; labels.appendChild(d); e.lb=d; ents.push(e);});
}
function fillSummary(){el("sVerts").textContent=num(MESH.total_vertices); el("sFaces").textContent=num(MESH.face_count); const b=bounds(); el("sMin").textContent=fmt(b.min); el("sMax").textContent=fmt(b.max); list.innerHTML="";
  ents.forEach(e=>{const bt=document.createElement("button"); bt.className="item"; bt.dataset.id=String(e.o.id); bt.dataset.label=String(e.o.label).toLowerCase();
  bt.innerHTML=`<div class=row><div><span class=sw style="background:${e.c}"></span>${esc(e.o.label)}</div><div class=muted>#${esc(String(e.o.id))}</div></div><div class=muted>center: ${fmt(e.o.center)}<br>dims: ${fmt(e.o.dimensions)}<br>vertices: ${num(e.o.vertex_count||0)}</div>`;
  bt.onclick=()=>pick(e,true); e.bt=bt; list.appendChild(bt);});
}
function bind(){
  el("q").oninput=()=>{const q=el("q").value.trim().toLowerCase(); ents.forEach(e=>{e.bt.style.display=!q||String(e.o.label).toLowerCase().includes(q)||String(e.o.id).includes(q)?"":"none"})};
  el("bReset").onclick=fitScene; el("bFit").onclick=()=>selected&&fitObj(selected);
  el("bMesh").onclick=()=>toggle("bMesh","mesh"); el("bBBox").onclick=()=>toggle("bBBox","bbox"); el("bLabel").onclick=()=>{showLabel=!showLabel; el("bLabel").classList.toggle("on",showLabel); updateLabels()};
  glCanvas.oncontextmenu=e=>e.preventDefault(); ovCanvas.oncontextmenu=e=>e.preventDefault();
  const down=e=>{ptr.m=e.button===2?"pan":"orb"; ptr.x=e.clientX; ptr.y=e.clientY}; glCanvas.onmousedown=down; ovCanvas.onmousedown=down;
  window.onmouseup=()=>{ptr.m=null}; window.onmousemove=e=>{if(!ptr.m) return; const dx=e.clientX-ptr.x,dy=e.clientY-ptr.y; ptr.x=e.clientX; ptr.y=e.clientY;
    if(ptr.m==="orb"){cam.yaw-=dx*.008; cam.pitch=clamp(cam.pitch+dy*.008,-1.45,1.45)} else {const b=basis(),s=cam.r*.0015; cam.t=add(cam.t,add(scale(b.r,-dx*s),scale(b.u,dy*s)))} request()};
  const wheel=e=>{e.preventDefault(); cam.r=clamp(cam.r*Math.exp(e.deltaY*.0012),.35,200); request()}; glCanvas.addEventListener("wheel",wheel,{passive:false}); ovCanvas.addEventListener("wheel",wheel,{passive:false});
  window.addEventListener("resize",resize);
}
function toggle(id,k){ if(k==="mesh") showMesh=!showMesh; else showBBox=!showBBox; el(id).classList.toggle("on",k==="mesh"?showMesh:showBBox); request(); }
function pick(e,fit){selected=e; ents.forEach(x=>{const on=x===e; x.bt.classList.toggle("sel",on);}); el("dLabel").textContent=e.o.label; el("dId").textContent=String(e.o.id); el("dCenter").textContent=fmt(e.o.center); el("dDims").textContent=fmt(e.o.dimensions); el("dMin").textContent=fmt(e.o.bbox_min); el("dMax").textContent=fmt(e.o.bbox_max); el("dVerts").textContent=num(e.o.vertex_count||0); el("dPts").textContent=num(e.p.length); updateLabels(); if(fit) fitObj(e); request();}
function resize(){const r=Math.min(window.devicePixelRatio||1,2),w=glCanvas.clientWidth||glCanvas.parentElement.clientWidth,h=glCanvas.clientHeight||glCanvas.parentElement.clientHeight; glCanvas.width=Math.max(1,Math.round(w*r)); glCanvas.height=Math.max(1,Math.round(h*r)); ovCanvas.width=Math.max(1,Math.round(w*r)); ovCanvas.height=Math.max(1,Math.round(h*r)); ovCanvas.style.width=w+"px"; ovCanvas.style.height=h+"px"; glCanvas.style.width=w+"px"; glCanvas.style.height=h+"px"; ctx2d.setTransform(r,0,0,r,0,0); gl.viewport(0,0,glCanvas.width,glCanvas.height); request();}
function request(){if(raf!==null) return; raf=requestAnimationFrame(draw)}
function draw(){raf=null; drawGL(); drawOverlay(); updateLabels();}
function drawGL(){gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT); if(!showMesh) return; const mvp=multiplyMat4(perspective(cam.fov,glCanvas.clientWidth/Math.max(glCanvas.clientHeight,1),cam.near,cam.far),viewMatrix()); const nrm=normalMatrix(); gl.useProgram(renderState.prog); gl.uniformMatrix4fv(renderState.uMvp,false,new Float32Array(mvp)); gl.uniformMatrix3fv(renderState.uN,false,new Float32Array(nrm)); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,renderState.idxBuf); gl.drawElements(gl.TRIANGLES,renderState.indexCount,gl.UNSIGNED_INT,0);}
function drawOverlay(){const w=ovCanvas.clientWidth,h=ovCanvas.clientHeight; ctx2d.clearRect(0,0,w,h); const b=basis(); drawGrid(b,w,h); drawAxes(b,w,h); if(selected) drawPoints(selected,b,w,h); if(showBBox&&selected) drawBBox(selected,b,w,h);}
function updateLabels(){ents.forEach(e=>e.lb.style.display="none"); if(!showLabel||!selected) return; const b=basis(),p=project(selected.o.center,b,glCanvas.clientWidth,glCanvas.clientHeight); if(!p.ok) return; selected.lb.style.display="block"; selected.lb.style.left=`${p.x}px`; selected.lb.style.top=`${p.y}px`;}
function drawPoints(e,b,w,h){ctx2d.save(); for(const p of e.p){const q=project(p,b,w,h); if(!q.ok) continue; const r=clamp(2.1-q.z*.04,.9,1.7); ctx2d.beginPath(); ctx2d.fillStyle="rgba(11,107,203,.7)"; ctx2d.arc(q.x,q.y,r,0,Math.PI*2); ctx2d.fill();} ctx2d.restore();}
function drawBBox(e,b,w,h){const pts=corners(e.o.bbox_min,e.o.bbox_max).map(p=>project(p,b,w,h)); if(pts.filter(x=>x.ok).length<2) return; ctx2d.save(); ctx2d.lineWidth=6; ctx2d.strokeStyle="rgba(255,255,255,.98)"; ED.forEach(([a,c])=>stroke2(pts[a],pts[c])); ctx2d.lineWidth=3; ctx2d.strokeStyle="#ff6b00"; ED.forEach(([a,c])=>stroke2(pts[a],pts[c])); ctx2d.fillStyle="#ff6b00"; pts.forEach(p=>{if(p.ok) ctx2d.fillRect(p.x-2.5,p.y-2.5,5,5)}); ctx2d.restore();}
function drawGrid(b,w,h){const bd=bounds(),z=bd.min[2],sx=bd.max[0]-bd.min[0],sy=bd.max[1]-bd.min[1],sp=Math.max(sx,sy,1),pad=sp*.18,x0=bd.min[0]-pad,x1=bd.max[0]+pad,y0=bd.min[1]-pad,y1=bd.max[1]+pad; ctx2d.save(); ctx2d.lineWidth=1; ctx2d.strokeStyle="rgba(182,195,209,.58)"; for(let i=0;i<=14;i++){const t=i/14,x=x0+(x1-x0)*t,y=y0+(y1-y0)*t; stroke([x,y0,z],[x,y1,z],b,w,h); stroke([x0,y,z],[x1,y,z],b,w,h)} ctx2d.restore();}
function drawAxes(b,w,h){const bd=bounds(),sp=Math.max(bd.max[0]-bd.min[0],bd.max[1]-bd.min[1],bd.max[2]-bd.min[2],1),o=[0,0,bd.min[2]]; ctx2d.save(); ctx2d.lineWidth=2.2; ctx2d.strokeStyle="#d14b45"; stroke(o,[o[0]+sp*.18,o[1],o[2]],b,w,h); ctx2d.strokeStyle="#2f8f4e"; stroke(o,[o[0],o[1]+sp*.18,o[2]],b,w,h); ctx2d.strokeStyle="#0b6bcb"; stroke(o,[o[0],o[1],o[2]+sp*.18],b,w,h); ctx2d.restore();}
function fitScene(){const bd=bounds(); const c=mid(bd.min,bd.max),s=sub(bd.max,bd.min),r=Math.max(Math.hypot(s[0],s[1],s[2])*.72,3); cam.t=c; cam.r=r*1.4; request();}
function fitObj(e){const c=e.o.center,s=e.o.dimensions,r=Math.max(Math.hypot(s[0],s[1],s[2])*.9,1.2); cam.t=c.slice(); cam.r=r*1.6; request();}
function bounds(){if(!OBJ.length) return {min:[-1,-1,-1],max:[1,1,1]}; const mn=OBJ[0].bbox_min.map(Number),mx=OBJ[0].bbox_max.map(Number); OBJ.forEach(o=>{for(let i=0;i<3;i++){mn[i]=Math.min(mn[i],Number(o.bbox_min[i])); mx[i]=Math.max(mx[i],Number(o.bbox_max[i]))}}); return {min:mn,max:mx};}
function basis(){const cp=Math.cos(cam.pitch),p=[cam.t[0]+cam.r*cp*Math.cos(cam.yaw),cam.t[1]+cam.r*cp*Math.sin(cam.yaw),cam.t[2]+cam.r*Math.sin(cam.pitch)],f=norm(sub(cam.t,p)); let r=norm(cross(f,[0,0,1])); if(len(r)<1e-6) r=[1,0,0]; const u=norm(cross(r,f)); return {p,f,r,u};}
function viewMatrix(){const b=basis(),p=b.p,r=b.r,u=b.u,f=b.f; return [r[0],u[0],-f[0],0,r[1],u[1],-f[1],0,r[2],u[2],-f[2],0,-dot(r,p),-dot(u,p),dot(f,p),1];}
function normalMatrix(){const b=basis(),r=b.r,u=b.u,f=b.f; return [r[0],u[0],-f[0],r[1],u[1],-f[1],r[2],u[2],-f[2]];}
function perspective(fov,aspect,near,far){const f=1/Math.tan(fov/2),nf=1/(near-far); return [f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0]}
function project(pt,b,w,h){const d=sub(pt,b.p),xc=dot(d,b.r),yc=dot(d,b.u),zc=dot(d,b.f); if(zc<=cam.near) return {ok:false,x:0,y:0,z:zc}; const foc=h/(2*Math.tan(cam.fov/2)); return {ok:true,z:zc,x:w/2+xc*foc/zc,y:h/2-yc*foc/zc};}
function corners(a,b){const[x0,y0,z0]=a,[x1,y1,z1]=b; return [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]]}
function stroke(a,b,ba,w,h){stroke2(project(a,ba,w,h),project(b,ba,w,h))} function stroke2(a,b){if(!a.ok||!b.ok) return; ctx2d.beginPath(); ctx2d.moveTo(a.x,a.y); ctx2d.lineTo(b.x,b.y); ctx2d.stroke();}
function multiplyMat4(a,b){const o=new Array(16).fill(0); for(let r=0;r<4;r++) for(let c=0;c<4;c++) for(let k=0;k<4;k++) o[c*4+r]+=a[k*4+r]*b[c*4+k]; return o;}
function add(a,b){return[a[0]+b[0],a[1]+b[1],a[2]+b[2]]} function sub(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]]} function scale(a,s){return[a[0]*s,a[1]*s,a[2]*s]}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]} function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
function len(a){return Math.hypot(a[0],a[1],a[2])} function norm(a){const l=len(a); return l<1e-8?[0,0,0]:[a[0]/l,a[1]/l,a[2]/l]} function mid(a,b){return[(a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2]}
function clamp(v,lo,hi){return Math.min(hi,Math.max(lo,v))} function col(i){return`hsl(${(i*47)%360},68%,50%)`} function esc(s){return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#39;")}
const ED=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
</script></body></html>""")


PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a WebGL ScanNet mesh viewer")
    p.add_argument("--scene_metadata", required=True)
    p.add_argument("--output", default="scene_webgl_viewer.html")
    p.add_argument("--title", default=None)
    p.add_argument("--mesh_path", default=None)
    p.add_argument("--segs_path", default=None)
    p.add_argument("--aggregation_path", default=None)
    p.add_argument("--meta_path", default=None)
    p.add_argument("--max_faces", type=int, default=50000)
    p.add_argument("--max_object_points", type=int, default=220)
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_html_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def infer_paths(meta_json: Path, scene_id: str, mesh: str | None, segs: str | None, agg: str | None, meta_txt: str | None) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    base = meta_json.parent

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


def parse_ply_header(path: Path) -> dict:
    with path.open("rb") as f:
        if f.readline().decode("ascii", errors="replace").strip() != "ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = None
        current = None
        vertex_count = face_count = None
        vertex_props: list[tuple[str, str]] = []
        face_prop = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} has an incomplete PLY header")
            s = line.decode("ascii", errors="replace").strip()
            if s == "end_header":
                data_start = f.tell()
                break
            if not s or s.startswith("comment"):
                continue
            parts = s.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                current = parts[1]
                if current == "vertex":
                    vertex_count = int(parts[2])
                elif current == "face":
                    face_count = int(parts[2])
            elif parts[0] == "property" and current == "vertex":
                if parts[1] == "list":
                    raise ValueError("list vertex properties are unsupported")
                vertex_props.append((parts[2], parts[1]))
            elif parts[0] == "property" and current == "face":
                face_prop = parts
    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format: {fmt}")
    if vertex_count is None or face_count is None:
        raise ValueError("PLY header missing vertex or face count")
    vertex_dtype = np.dtype([(name, PLY_TYPES[type_name]) for name, type_name in vertex_props])
    if not face_prop or face_prop[:4] != ["property", "list", "uchar", "int"]:
        raise ValueError("unsupported face property layout")
    face_dtype = np.dtype([("count", "u1"), ("vertex_indices", "<i4", (3,))])
    return {"vertex_count": vertex_count, "face_count": face_count, "vertex_dtype": vertex_dtype, "face_dtype": face_dtype, "data_start": data_start}


def load_axis_alignment(meta_path: Path | None) -> np.ndarray:
    if meta_path is None or not meta_path.exists():
        return np.eye(4, dtype=np.float32)
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if "axisAlignment" in line:
            values = [float(x) for x in line.split("=")[1].strip().split()]
            return np.array(values, dtype=np.float32).reshape(4, 4)
    return np.eye(4, dtype=np.float32)


def load_mesh(mesh_path: Path, meta_path: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hdr = parse_ply_header(mesh_path)
    with mesh_path.open("rb") as f:
        f.seek(hdr["data_start"])
        verts = np.fromfile(f, dtype=hdr["vertex_dtype"], count=hdr["vertex_count"])
        faces = np.fromfile(f, dtype=hdr["face_dtype"], count=hdr["face_count"])
    if not np.all(faces["count"] == 3):
        faces = faces[faces["count"] == 3]

    positions = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)
    colors = np.stack([verts["red"], verts["green"], verts["blue"]], axis=1).astype(np.float32) / 255.0
    if {"nx", "ny", "nz"}.issubset(verts.dtype.names or []):
        normals = np.stack([verts["nx"], verts["ny"], verts["nz"]], axis=1).astype(np.float32)
    else:
        normals = np.zeros_like(positions, dtype=np.float32)

    M = load_axis_alignment(meta_path)
    R = M[:3, :3].astype(np.float32)
    t = M[:3, 3].astype(np.float32)
    positions = positions @ R.T + t
    normals = normals @ R.T
    normal_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_len, 1e-6)
    return positions, colors, normals, faces["vertex_indices"].astype(np.int32)


def load_object_vertex_ids(scene: dict, segs_path: Path, aggregation_path: Path) -> np.ndarray:
    seg_indices = np.array(load_json(segs_path)["segIndices"], dtype=np.int32)
    max_seg = int(seg_indices.max(initial=-1))
    seg_to_obj = np.full(max_seg + 1, -1, dtype=np.int32)
    for group in load_json(aggregation_path).get("segGroups", []):
        oid = int(group["objectId"])
        for seg in group.get("segments", []):
            seg_to_obj[int(seg)] = oid
    return seg_to_obj[seg_indices]


def sample_object_points(scene: dict, positions: np.ndarray, obj_ids: np.ndarray, max_points: int) -> dict[str, list[list[float]]]:
    out: dict[str, list[list[float]]] = {}
    for obj in scene.get("objects", []):
        oid = int(obj["id"])
        idx = np.flatnonzero(obj_ids == oid)
        if idx.size == 0:
            out[str(oid)] = []
            continue
        stride = max(1, math.ceil(idx.size / max(max_points, 1)))
        pts = positions[idx[::stride][:max_points]]
        out[str(oid)] = np.round(pts, 4).tolist()
    return out


def sample_mesh(positions: np.ndarray, colors: np.ndarray, normals: np.ndarray, faces: np.ndarray, max_faces: int) -> dict:
    face_stride = max(1, math.ceil(len(faces) / max(max_faces, 1)))
    sampled_faces = faces[::face_stride]
    unique_idx, inverse = np.unique(sampled_faces.reshape(-1), return_inverse=True)
    mesh_positions = np.round(positions[unique_idx], 4).astype(np.float32).reshape(-1)
    mesh_colors = np.round(colors[unique_idx], 4).astype(np.float32).reshape(-1)
    mesh_normals = np.round(normals[unique_idx], 4).astype(np.float32).reshape(-1)
    return {
        "positions": mesh_positions.tolist(),
        "colors": mesh_colors.tolist(),
        "normals": mesh_normals.tolist(),
        "indices": inverse.astype(np.uint32).tolist(),
        "total_vertices": int(len(positions)),
        "face_count": int(len(sampled_faces)),
    }


def main() -> None:
    a = parse_args()
    scene_path = Path(a.scene_metadata)
    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scene = load_json(scene_path)
    scene_id = str(scene.get("scene_id") or scene_path.stem)
    mesh_path, segs_path, agg_path, meta_path = infer_paths(scene_path, scene_id, a.mesh_path, a.segs_path, a.aggregation_path, a.meta_path)
    if not (mesh_path and segs_path and agg_path):
        raise FileNotFoundError("mesh/segs/aggregation files are required for the WebGL viewer")

    positions, colors, normals, faces = load_mesh(mesh_path, meta_path)
    obj_ids = load_object_vertex_ids(scene, segs_path, agg_path)
    if len(obj_ids) != len(positions):
        raise ValueError("segIndices length does not match mesh vertex count")

    object_point_samples = sample_object_points(scene, positions, obj_ids, a.max_object_points)
    mesh_bundle = sample_mesh(positions, colors, normals, faces, a.max_faces)

    scene_bundle = dict(scene)
    scene_bundle["object_point_samples"] = object_point_samples
    scene_type = "unknown"
    if meta_path and meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("sceneType"):
                scene_type = line.split("=")[1].strip()
                break

    html = PAGE.safe_substitute(
        title=a.title or f"{scene_id} WebGL mesh viewer",
        scene_id=scene_id,
        scene_type=scene_type,
        object_count=len(scene.get("objects") or []),
        scene_json=dump_html_json(scene_bundle),
        mesh_json=dump_html_json(mesh_bundle),
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote WebGL viewer to {out_path}")
    print(f"embedded sampled mesh: {mesh_bundle['face_count']} faces from {mesh_path.name}")


if __name__ == "__main__":
    main()
