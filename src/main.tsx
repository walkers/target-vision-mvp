import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

type Point = { x: number; y: number };
type Marker = { x: number; y: number; r: number; score: number };

const TARGET_W = 900;
const TARGET_H = 1200;
const MAX_POINTS = 4;

function clamp(n: number, min: number, max: number) { return Math.max(min, Math.min(max, n)); }
function encodeSignal(obj: unknown): string { return btoa(unescape(encodeURIComponent(JSON.stringify(obj)))); }
function decodeSignal<T>(text: string): T { return JSON.parse(decodeURIComponent(escape(atob(text.trim())))); }
async function copyText(text: string) { await navigator.clipboard.writeText(text); }

function orderCorners(points: Point[]): Point[] {
  const sorted = [...points].sort((a, b) => a.y - b.y);
  const top = sorted.slice(0, 2).sort((a, b) => a.x - b.x);
  const bottom = sorted.slice(2, 4).sort((a, b) => a.x - b.x);
  return [top[0], top[1], bottom[1], bottom[0]];
}

function solveHomography(src: Point[], dst: Point[]): number[] {
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = src[i];
    const u = dst[i].x;
    const v = dst[i].y;
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  for (let i = 0; i < 8; i++) {
    let maxRow = i;
    for (let r = i + 1; r < 8; r++) if (Math.abs(A[r][i]) > Math.abs(A[maxRow][i])) maxRow = r;
    [A[i], A[maxRow]] = [A[maxRow], A[i]]; [b[i], b[maxRow]] = [b[maxRow], b[i]];
    const pivot = A[i][i] || 1e-12;
    for (let c = i; c < 8; c++) A[i][c] /= pivot;
    b[i] /= pivot;
    for (let r = 0; r < 8; r++) {
      if (r === i) continue;
      const factor = A[r][i];
      for (let c = i; c < 8; c++) A[r][c] -= factor * A[i][c];
      b[r] -= factor * b[i];
    }
  }
  return [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], 1];
}

function invertHomography(m: number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = c * h - b * i, C = b * f - c * e;
  const D = f * g - d * i, E = a * i - c * g, F = c * d - a * f;
  const G = d * h - e * g, H = b * g - a * h, I = a * e - b * d;
  const det = a * A + b * D + c * G || 1e-12;
  return [A / det, B / det, C / det, D / det, E / det, F / det, G / det, H / det, I / det];
}

function warpVideoToCanvas(video: HTMLVideoElement, corners: Point[], canvas: HTMLCanvasElement) {
  canvas.width = TARGET_W; canvas.height = TARGET_H;
  const ctx = canvas.getContext('2d', { willReadFrequently: true }); if (!ctx) return;
  const ordered = orderCorners(corners);
  const dst = [{x:0,y:0},{x:TARGET_W-1,y:0},{x:TARGET_W-1,y:TARGET_H-1},{x:0,y:TARGET_H-1}];
  const inv = invertHomography(solveHomography(ordered, dst));
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = video.videoWidth; srcCanvas.height = video.videoHeight;
  const sctx = srcCanvas.getContext('2d', { willReadFrequently: true })!;
  sctx.drawImage(video, 0, 0);
  const srcImg = sctx.getImageData(0, 0, srcCanvas.width, srcCanvas.height);
  const out = ctx.createImageData(TARGET_W, TARGET_H);
  for (let y = 0; y < TARGET_H; y++) for (let x = 0; x < TARGET_W; x++) {
    const den = inv[6] * x + inv[7] * y + inv[8];
    const sx = (inv[0] * x + inv[1] * y + inv[2]) / den;
    const sy = (inv[3] * x + inv[4] * y + inv[5]) / den;
    const ix = clamp(Math.round(sx), 0, srcCanvas.width - 1);
    const iy = clamp(Math.round(sy), 0, srcCanvas.height - 1);
    const si = (iy * srcCanvas.width + ix) * 4;
    const oi = (y * TARGET_W + x) * 4;
    out.data[oi] = srcImg.data[si]; out.data[oi+1] = srcImg.data[si+1]; out.data[oi+2] = srcImg.data[si+2]; out.data[oi+3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

function grayAt(data: Uint8ClampedArray, x: number, y: number, w: number): number {
  const i = (y*w+x)*4; return 0.299*data[i]+0.587*data[i+1]+0.114*data[i+2];
}
function findBestTranslation(baseline: ImageData, current: ImageData, maxShift=18, step=3) {
  let best = { dx: 0, dy: 0, score: Number.POSITIVE_INFINITY };
  const sampleStep=18, margin=maxShift+8;
  for (let dy=-maxShift; dy<=maxShift; dy+=step) for (let dx=-maxShift; dx<=maxShift; dx+=step) {
    let score=0, count=0;
    for (let y=margin; y<TARGET_H-margin; y+=sampleStep) for (let x=margin; x<TARGET_W-margin; x+=sampleStep) {
      score += Math.abs(grayAt(baseline.data,x,y,TARGET_W) - grayAt(current.data,x+dx,y+dy,TARGET_W)); count++;
    }
    score /= Math.max(1,count); if (score < best.score) best = {dx,dy,score};
  }
  return best;
}
function shiftedGray(data: Uint8ClampedArray, x: number, y: number, dx: number, dy: number) {
  return grayAt(data, clamp(x+dx,0,TARGET_W-1), clamp(y+dy,0,TARGET_H-1), TARGET_W);
}

function enhanceAndDetect(baseline: HTMLCanvasElement, current: HTMLCanvasElement, output: HTMLCanvasElement) {
  output.width=TARGET_W; output.height=TARGET_H;
  const bctx=baseline.getContext('2d',{willReadFrequently:true})!, cctx=current.getContext('2d',{willReadFrequently:true})!, octx=output.getContext('2d',{willReadFrequently:true})!;
  const b=bctx.getImageData(0,0,TARGET_W,TARGET_H), c=cctx.getImageData(0,0,TARGET_W,TARGET_H);
  const shift=findBestTranslation(b,c), out=octx.createImageData(TARGET_W,TARGET_H), mask=new Uint8Array(TARGET_W*TARGET_H);
  for (let y=0;y<TARGET_H;y++) for (let x=0;x<TARGET_W;x++) {
    const p=y*TARGET_W+x, i=p*4, bg=grayAt(b.data,x,y,TARGET_W), cg=shiftedGray(c.data,x,y,shift.dx,shift.dy), darkening=bg-cg, hi=cg<140?0:255;
    out.data[i]=hi; out.data[i+1]=hi; out.data[i+2]=hi; out.data[i+3]=255;
    if (darkening>26 && cg<190) mask[p]=1;
  }
  const visited=new Uint8Array(TARGET_W*TARGET_H), markers: Marker[]=[], qx=new Int32Array(24000), qy=new Int32Array(24000);
  for (let y=2;y<TARGET_H-2;y++) for (let x=2;x<TARGET_W-2;x++) {
    const start=y*TARGET_W+x; if (!mask[start]||visited[start]) continue;
    let head=0, tail=0, count=0, sumX=0, sumY=0, minX=x, maxX=x, minY=y, maxY=y;
    qx[tail]=x; qy[tail]=y; tail++; visited[start]=1;
    while (head<tail && tail<qx.length) {
      const cx=qx[head], cy=qy[head]; head++; count++; sumX+=cx; sumY+=cy; minX=Math.min(minX,cx); maxX=Math.max(maxX,cx); minY=Math.min(minY,cy); maxY=Math.max(maxY,cy);
      for (let yy=-1;yy<=1;yy++) for (let xx=-1;xx<=1;xx++) {
        if (!xx&&!yy) continue; const nx=cx+xx, ny=cy+yy, ni=ny*TARGET_W+nx;
        if (nx<=1||ny<=1||nx>=TARGET_W-2||ny>=TARGET_H-2) continue;
        if (mask[ni]&&!visited[ni]) { visited[ni]=1; qx[tail]=nx; qy[tail]=ny; tail++; }
      }
    }
    const w=maxX-minX+1, h=maxY-minY+1, aspect=Math.max(w,h)/Math.max(1,Math.min(w,h)), density=count/Math.max(1,w*h);
    if (count>=8 && count<=1600 && aspect<3 && density>.18 && w<75 && h<75) markers.push({x:sumX/count,y:sumY/count,r:Math.max(12,Math.min(34,Math.sqrt(count)*1.7)),score:count});
  }
  markers.sort((a,b)=>b.score-a.score); octx.putImageData(out,0,0);
  octx.lineWidth=7; octx.strokeStyle='#ff0044'; octx.fillStyle='#ff0044';
  markers.slice(0,30).forEach((m,idx)=>{ octx.beginPath(); octx.arc(m.x,m.y,m.r,0,Math.PI*2); octx.stroke(); octx.font='bold 32px system-ui'; octx.fillText(String(idx+1),m.x+m.r+6,m.y-m.r-6); });
  octx.fillStyle='rgba(255,255,255,.9)'; octx.font='bold 24px system-ui'; octx.fillText(`stabilized shift: ${shift.dx}, ${shift.dy}`,18,TARGET_H-24);
  return {markers, shift};
}

function SenderPage() {
  const videoRef=useRef<HTMLVideoElement>(null), pcRef=useRef<RTCPeerConnection|null>(null), streamRef=useRef<MediaStream|null>(null);
  const [status,setStatus]=useState('Start camera, then create offer.'), [offer,setOffer]=useState(''), [answerText,setAnswerText]=useState('');
  async function startCamera(){ try{ const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1920},height:{ideal:1080},frameRate:{ideal:30}},audio:false}); streamRef.current=stream; if(videoRef.current){videoRef.current.srcObject=stream; await videoRef.current.play();} setStatus('Camera ready. Create offer.'); }catch(err){ setStatus(`Camera failed: ${err instanceof Error ? err.message : String(err)}`); }}
  async function createOffer(){ if(!streamRef.current){setStatus('Start camera first.');return;} const pc=new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'}]}); pcRef.current=pc; streamRef.current.getTracks().forEach(t=>pc.addTrack(t,streamRef.current!)); pc.onconnectionstatechange=()=>setStatus(`Connection: ${pc.connectionState}`); await pc.setLocalDescription(await pc.createOffer()); setStatus('Gathering connection candidates...'); await new Promise<void>(resolve=>{ if(pc.iceGatheringState==='complete') resolve(); pc.onicegatheringstatechange=()=>{ if(pc.iceGatheringState==='complete') resolve(); }; setTimeout(resolve,2500); }); setOffer(encodeSignal(pc.localDescription)); setStatus('Offer ready. Copy it to Viewer phone.'); }
  async function acceptAnswer(){ try{ const pc=pcRef.current; if(!pc){setStatus('Create offer first.');return;} await pc.setRemoteDescription(decodeSignal<RTCSessionDescriptionInit>(answerText)); setStatus('Answer accepted. Video should connect.'); }catch(err){ setStatus(`Answer failed: ${err instanceof Error ? err.message : String(err)}`); }}
  return <main><header><h1>Camera Sender</h1><p>{status}</p></header><section className="panel cameraPanel"><video ref={videoRef} playsInline muted autoPlay /></section><section className="controls"><button onClick={startCamera}>Start Camera</button><button onClick={createOffer}>Create Offer</button></section><section className="panel outputPanel"><h2>1. Offer for Viewer</h2><textarea readOnly value={offer} placeholder="Offer appears here..." /><button disabled={!offer} onClick={()=>copyText(offer)}>Copy Offer</button></section><section className="panel outputPanel"><h2>2. Paste Viewer Answer</h2><textarea value={answerText} onChange={e=>setAnswerText(e.target.value)} placeholder="Paste answer here..." /><button onClick={acceptAnswer}>Accept Answer</button></section></main>;
}

function ViewerPage() {
  const videoRef=useRef<HTMLVideoElement>(null), overlayRef=useRef<HTMLCanvasElement>(null), currentRef=useRef<HTMLCanvasElement>(document.createElement('canvas')), baselineRef=useRef<HTMLCanvasElement|null>(null), outputRef=useRef<HTMLCanvasElement>(null);
  const [status,setStatus]=useState('Paste Sender offer, then connect.'), [offerText,setOfferText]=useState(''), [answerText,setAnswerText]=useState(''), [points,setPoints]=useState<Point[]>([]), [markers,setMarkers]=useState<Marker[]>([]), [hasBaseline,setHasBaseline]=useState(false), [lastShift,setLastShift]=useState<{dx:number;dy:number;score:number}|null>(null);
  useEffect(()=>{ const video=videoRef.current, canvas=overlayRef.current; if(!video||!canvas)return; const rect=video.getBoundingClientRect(); canvas.width=rect.width; canvas.height=rect.height; const ctx=canvas.getContext('2d')!; ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle='#00ff88'; ctx.strokeStyle='#00ff88'; ctx.lineWidth=3; ctx.font='bold 18px system-ui'; points.forEach((p,idx)=>{ const sx=(p.x/video.videoWidth)*canvas.width, sy=(p.y/video.videoHeight)*canvas.height; ctx.beginPath(); ctx.arc(sx,sy,10,0,Math.PI*2); ctx.fill(); ctx.fillText(String(idx+1),sx+12,sy-12); }); if(points.length===4){ const ordered=orderCorners(points); ctx.beginPath(); ordered.forEach((p,idx)=>{ const sx=(p.x/video.videoWidth)*canvas.width, sy=(p.y/video.videoHeight)*canvas.height; if(idx===0)ctx.moveTo(sx,sy); else ctx.lineTo(sx,sy); }); ctx.closePath(); ctx.stroke(); }},[points]);
  async function connectToSender(){ try{ const offer=decodeSignal<RTCSessionDescriptionInit>(offerText); const pc=new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'}]}); pc.ontrack=async e=>{ const [stream]=e.streams; if(videoRef.current){ videoRef.current.srcObject=stream; await videoRef.current.play(); } setStatus('Remote camera connected. Tap 4 target corners.'); }; pc.onconnectionstatechange=()=>setStatus(`Connection: ${pc.connectionState}`); await pc.setRemoteDescription(offer); await pc.setLocalDescription(await pc.createAnswer()); setStatus('Gathering connection candidates...'); await new Promise<void>(resolve=>{ if(pc.iceGatheringState==='complete') resolve(); pc.onicegatheringstatechange=()=>{ if(pc.iceGatheringState==='complete') resolve(); }; setTimeout(resolve,2500); }); setAnswerText(encodeSignal(pc.localDescription)); setStatus('Answer ready. Copy it back to Sender.'); }catch(err){ setStatus(`Connect failed: ${err instanceof Error ? err.message : String(err)}`); }}
  function onVideoTap(e:React.PointerEvent<HTMLDivElement>){ const video=videoRef.current; if(!video||!video.videoWidth||points.length>=MAX_POINTS)return; const rect=(e.currentTarget.querySelector('video') as HTMLVideoElement).getBoundingClientRect(); setPoints([...points,{x:((e.clientX-rect.left)/rect.width)*video.videoWidth,y:((e.clientY-rect.top)/rect.height)*video.videoHeight}]); }
  function resetCorners(){ setPoints([]); setMarkers([]); setHasBaseline(false); setLastShift(null); baselineRef.current=null; setStatus('Corners reset. Tap 4 target corners.'); }
  function captureBaseline(){ const video=videoRef.current; if(!video||points.length!==4||!video.videoWidth){setStatus('Need remote video and 4 target corners first.');return;} const canvas=document.createElement('canvas'); warpVideoToCanvas(video,points,canvas); baselineRef.current=canvas; setHasBaseline(true); setMarkers([]); setLastShift(null); setStatus('Baseline captured. Mark/fire target, then Detect Shot.'); }
  function detect(){ const video=videoRef.current, baseline=baselineRef.current, output=outputRef.current; if(!video||!baseline||!output||points.length!==4){setStatus('Need video, corners, and baseline.');return;} warpVideoToCanvas(video,points,currentRef.current); const result=enhanceAndDetect(baseline,currentRef.current,output); setMarkers(result.markers); setLastShift(result.shift); setStatus(`Detected ${result.markers.length} candidate mark${result.markers.length===1?'':'s'}.`); }
  return <main><header><h1>Target Viewer</h1><p>{status}</p></header><section className="panel outputPanel"><h2>1. Paste Sender Offer</h2><textarea value={offerText} onChange={e=>setOfferText(e.target.value)} placeholder="Paste offer from camera phone..." /><button onClick={connectToSender}>Connect to Sender</button><h2>2. Viewer Answer</h2><textarea readOnly value={answerText} placeholder="Answer appears here..." /><button disabled={!answerText} onClick={()=>copyText(answerText)}>Copy Answer</button></section><section className="panel cameraPanel" onPointerDown={onVideoTap}><video ref={videoRef} playsInline muted autoPlay /><canvas ref={overlayRef} className="overlay" /></section><section className="controls"><button onClick={resetCorners}>Reset Corners</button><button disabled={points.length!==4} onClick={captureBaseline}>Capture Baseline</button><button disabled={!hasBaseline} onClick={detect}>Detect Shot</button></section><section className="readout"><div>Corner points: {points.length}/4</div><div>Candidate marks: {markers.length}</div></section><section className="readout"><div>Stabilization: {lastShift?`${lastShift.dx}, ${lastShift.dy}`:'not run yet'}</div><div>Match score: {lastShift?lastShift.score.toFixed(1):'—'}</div></section><section className="panel outputPanel"><h2>Enhanced Target</h2><canvas ref={outputRef} /></section></main>;
}

function HomePage(){ const base=window.location.origin; return <main><header><h1>Target Vision Two-Phone Test</h1><p>Open Sender on the camera iPhone and Viewer on the shooter iPhone.</p></header><section className="panel outputPanel"><h2>Open these URLs</h2><a className="bigLink" href="/sender">{base}/sender</a><a className="bigLink" href="/viewer">{base}/viewer</a></section><footer>This build uses manual WebRTC offer/answer copy-paste. It is intentionally crude so we can prove the two-phone video architecture without a backend.</footer></main>; }
function App(){ const path=window.location.pathname.toLowerCase(); if(path.startsWith('/sender'))return <SenderPage/>; if(path.startsWith('/viewer'))return <ViewerPage/>; return <HomePage/>; }
createRoot(document.getElementById('root')!).render(<App />);
