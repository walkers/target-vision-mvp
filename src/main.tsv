import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

type Point = { x: number; y: number };
type Marker = { x: number; y: number; r: number; score: number };

const TARGET_W = 900;
const TARGET_H = 1200;
const MAX_POINTS = 4;

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function orderCorners(points: Point[]): Point[] {
  // Returns TL, TR, BR, BL. Good enough for user-tapped target corners.
  const sorted = [...points].sort((a, b) => a.y - b.y);
  const top = sorted.slice(0, 2).sort((a, b) => a.x - b.x);
  const bottom = sorted.slice(2, 4).sort((a, b) => a.x - b.x);
  return [top[0], top[1], bottom[1], bottom[0]];
}

function solveHomography(src: Point[], dst: Point[]): number[] {
  // Solves 8 unknowns for projective transform mapping src -> dst.
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = src[i];
    const u = dst[i].x;
    const v = dst[i].y;
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }

  // Gaussian elimination.
  for (let i = 0; i < 8; i++) {
    let maxRow = i;
    for (let r = i + 1; r < 8; r++) {
      if (Math.abs(A[r][i]) > Math.abs(A[maxRow][i])) maxRow = r;
    }
    [A[i], A[maxRow]] = [A[maxRow], A[i]];
    [b[i], b[maxRow]] = [b[maxRow], b[i]];

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
  const A = e * i - f * h;
  const B = c * h - b * i;
  const C = b * f - c * e;
  const D = f * g - d * i;
  const E = a * i - c * g;
  const F = c * d - a * f;
  const G = d * h - e * g;
  const H = b * g - a * h;
  const I = a * e - b * d;
  const det = a * A + b * D + c * G || 1e-12;
  return [A / det, B / det, C / det, D / det, E / det, F / det, G / det, H / det, I / det];
}

function warpToCanvas(video: HTMLVideoElement, corners: Point[], canvas: HTMLCanvasElement) {
  canvas.width = TARGET_W;
  canvas.height = TARGET_H;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return;
  const ordered = orderCorners(corners);
  const dst = [{ x: 0, y: 0 }, { x: TARGET_W - 1, y: 0 }, { x: TARGET_W - 1, y: TARGET_H - 1 }, { x: 0, y: TARGET_H - 1 }];
  const h = solveHomography(ordered, dst);
  const inv = invertHomography(h);

  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = video.videoWidth;
  srcCanvas.height = video.videoHeight;
  const sctx = srcCanvas.getContext('2d', { willReadFrequently: true })!;
  sctx.drawImage(video, 0, 0);
  const srcImg = sctx.getImageData(0, 0, srcCanvas.width, srcCanvas.height);
  const out = ctx.createImageData(TARGET_W, TARGET_H);

  for (let y = 0; y < TARGET_H; y++) {
    for (let x = 0; x < TARGET_W; x++) {
      const den = inv[6] * x + inv[7] * y + inv[8];
      const sx = (inv[0] * x + inv[1] * y + inv[2]) / den;
      const sy = (inv[3] * x + inv[4] * y + inv[5]) / den;
      const ix = clamp(Math.round(sx), 0, srcCanvas.width - 1);
      const iy = clamp(Math.round(sy), 0, srcCanvas.height - 1);
      const si = (iy * srcCanvas.width + ix) * 4;
      const oi = (y * TARGET_W + x) * 4;
      out.data[oi] = srcImg.data[si];
      out.data[oi + 1] = srcImg.data[si + 1];
      out.data[oi + 2] = srcImg.data[si + 2];
      out.data[oi + 3] = 255;
    }
  }
  ctx.putImageData(out, 0, 0);
}

function enhanceAndDetect(baseline: HTMLCanvasElement, current: HTMLCanvasElement, output: HTMLCanvasElement): Marker[] {
  output.width = TARGET_W;
  output.height = TARGET_H;
  const bctx = baseline.getContext('2d', { willReadFrequently: true })!;
  const cctx = current.getContext('2d', { willReadFrequently: true })!;
  const octx = output.getContext('2d', { willReadFrequently: true })!;
  const b = bctx.getImageData(0, 0, TARGET_W, TARGET_H);
  const c = cctx.getImageData(0, 0, TARGET_W, TARGET_H);
  const out = octx.createImageData(TARGET_W, TARGET_H);

  const mask = new Uint8Array(TARGET_W * TARGET_H);
  for (let p = 0; p < TARGET_W * TARGET_H; p++) {
    const i = p * 4;
    const bg = 0.299 * b.data[i] + 0.587 * b.data[i + 1] + 0.114 * b.data[i + 2];
    const cg = 0.299 * c.data[i] + 0.587 * c.data[i + 1] + 0.114 * c.data[i + 2];
    const darkening = bg - cg;
    const hi = cg < 140 ? 0 : 255;
    out.data[i] = hi;
    out.data[i + 1] = hi;
    out.data[i + 2] = hi;
    out.data[i + 3] = 255;
    if (darkening > 22 && cg < 190) mask[p] = 1;
  }

  const visited = new Uint8Array(TARGET_W * TARGET_H);
  const markers: Marker[] = [];
  const qx = new Int32Array(20000);
  const qy = new Int32Array(20000);

  for (let y = 2; y < TARGET_H - 2; y++) {
    for (let x = 2; x < TARGET_W - 2; x++) {
      const start = y * TARGET_W + x;
      if (!mask[start] || visited[start]) continue;
      let head = 0, tail = 0;
      qx[tail] = x; qy[tail] = y; tail++;
      visited[start] = 1;
      let count = 0, sumX = 0, sumY = 0, minX = x, maxX = x, minY = y, maxY = y;
      while (head < tail && tail < qx.length) {
        const cx = qx[head], cy = qy[head]; head++;
        count++; sumX += cx; sumY += cy;
        minX = Math.min(minX, cx); maxX = Math.max(maxX, cx);
        minY = Math.min(minY, cy); maxY = Math.max(maxY, cy);
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const nx = cx + dx, ny = cy + dy, ni = ny * TARGET_W + nx;
            if (nx <= 1 || ny <= 1 || nx >= TARGET_W - 2 || ny >= TARGET_H - 2) continue;
            if (mask[ni] && !visited[ni]) {
              visited[ni] = 1; qx[tail] = nx; qy[tail] = ny; tail++;
            }
          }
        }
      }
      const w = maxX - minX + 1;
      const h = maxY - minY + 1;
      const aspect = Math.max(w, h) / Math.max(1, Math.min(w, h));
      if (count >= 8 && count <= 1500 && aspect < 3.0 && w < 70 && h < 70) {
        markers.push({ x: sumX / count, y: sumY / count, r: Math.max(12, Math.min(34, Math.sqrt(count) * 1.7)), score: count });
      }
    }
  }

  octx.putImageData(out, 0, 0);
  octx.lineWidth = 6;
  octx.strokeStyle = '#ff0044';
  octx.fillStyle = '#ff0044';
  markers.slice(0, 30).forEach((m, idx) => {
    octx.beginPath();
    octx.arc(m.x, m.y, m.r, 0, Math.PI * 2);
    octx.stroke();
    octx.font = 'bold 28px system-ui';
    octx.fillText(String(idx + 1), m.x + m.r + 4, m.y - m.r - 4);
  });
  return markers;
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const currentRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));
  const baselineRef = useRef<HTMLCanvasElement | null>(null);
  const outputRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [status, setStatus] = useState('Tap Start Camera');
  const [markers, setMarkers] = useState<Marker[]>([]);
  const [hasBaseline, setHasBaseline] = useState(false);

  useEffect(() => {
    const draw = () => {
      const video = videoRef.current;
      const canvas = overlayRef.current;
      if (!video || !canvas) return;
      const rect = video.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      const ctx = canvas.getContext('2d')!;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#00ff88';
      ctx.strokeStyle = '#00ff88';
      ctx.lineWidth = 3;
      points.forEach((p, idx) => {
        const sx = p.x / video.videoWidth * canvas.width;
        const sy = p.y / video.videoHeight * canvas.height;
        ctx.beginPath(); ctx.arc(sx, sy, 10, 0, Math.PI * 2); ctx.fill();
        ctx.fillText(String(idx + 1), sx + 12, sy - 12);
      });
      if (points.length === 4) {
        const ordered = orderCorners(points);
        ctx.beginPath();
        ordered.forEach((p, idx) => {
          const sx = p.x / video.videoWidth * canvas.width;
          const sy = p.y / video.videoHeight * canvas.height;
          if (idx === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
        });
        ctx.closePath(); ctx.stroke();
      }
    };
    draw();
  }, [points]);

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false
      });
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();
      setStatus('Camera ready. Tap 4 target corners.');
    } catch (err) {
      setStatus(`Camera failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  function onVideoTap(e: React.PointerEvent<HTMLDivElement>) {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    if (points.length >= MAX_POINTS) return;
    const rect = (e.currentTarget.querySelector('video') as HTMLVideoElement).getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * video.videoWidth;
    const y = (e.clientY - rect.top) / rect.height * video.videoHeight;
    setPoints([...points, { x, y }]);
  }

  function captureBaseline() {
    const video = videoRef.current;
    if (!video || points.length !== 4) return setStatus('Need 4 target corners first.');
    const canvas = document.createElement('canvas');
    warpToCanvas(video, points, canvas);
    baselineRef.current = canvas;
    setHasBaseline(true);
    setMarkers([]);
    setStatus('Baseline captured. Fire/mark target, then tap Detect.');
  }

  function detect() {
    const video = videoRef.current;
    const baseline = baselineRef.current;
    const output = outputRef.current;
    if (!video || !baseline || !output || points.length !== 4) return setStatus('Need camera, corners, and baseline.');
    warpToCanvas(video, points, currentRef.current);
    const found = enhanceAndDetect(baseline, currentRef.current, output);
    setMarkers(found);
    setStatus(`Detected ${found.length} candidate mark${found.length === 1 ? '' : 's'}.`);
  }

  function resetCorners() {
    setPoints([]); setMarkers([]); setHasBaseline(false); baselineRef.current = null;
    setStatus('Corners reset. Tap 4 target corners.');
  }

  return (
    <main>
      <header>
        <h1>Target Vision MVP</h1>
        <p>{status}</p>
      </header>

      <section className="panel cameraPanel" onPointerDown={onVideoTap}>
        <video ref={videoRef} playsInline muted />
        <canvas ref={overlayRef} className="overlay" />
      </section>

      <section className="controls">
        <button onClick={startCamera}>Start Camera</button>
        <button onClick={resetCorners}>Reset Corners</button>
        <button disabled={points.length !== 4} onClick={captureBaseline}>Capture Baseline</button>
        <button disabled={!hasBaseline} onClick={detect}>Detect Shot</button>
      </section>

      <section className="readout">
        <div>Corner points: {points.length}/4</div>
        <div>Candidate marks: {markers.length}</div>
      </section>

      <section className="panel outputPanel">
        <h2>Enhanced Target</h2>
        <canvas ref={outputRef} />
      </section>

      <footer>
        MVP A: manual detection only. Use at 3 yards first. Keep the phone steady between baseline and detect.
      </footer>
    </main>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
