/**
 * 2D overlay drawn over the WebGPU canvas: cursor with brush-radius ticks,
 * clone markers, stroke line, example edge, edge preview and the colour
 * selector — the immediate-mode GL drawing at the end of GeneralPaint::draw.
 */
import { hsvToRgb, RGB } from '../paint/hsv';
import type { Vec2 } from '../paint/stroke';
import type { Mode } from './app';

export interface OverlayState {
  dpr: number;
  mouse: Vec2;
  mouseInside: boolean;
  brushRadiusPx: number;
  toScreen(p: Vec2): Vec2;
  mode: Mode;
  clonePoint: Vec2 | null;
  cloneSource: Vec2 | null;
  strokeLine: Vec2[] | null;
  exampleEdge: Vec2[] | null;
  edgePreview: { perp: ImageData; tan: ImageData } | null;
  edgePreviewScale: number;
  colorSelector: { hsv: RGB; valueMode: boolean; radius: number } | null;
}

const css = (c: RGB) => `rgb(${c.map((v) => Math.round(Math.max(0, Math.min(1, v)) * 255)).join(',')})`;

function crosshair(ctx: CanvasRenderingContext2D, [x, y]: Vec2, r = 5, color = '#fff') {
  ctx.strokeStyle = 'rgba(0,0,0,0.6)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(x - r, y);
  ctx.lineTo(x + r, y);
  ctx.moveTo(x, y - r);
  ctx.lineTo(x, y + r);
  ctx.stroke();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.stroke();
}

let wheelCache: { key: string; canvas: HTMLCanvasElement } | null = null;

/** Hue/saturation disc at full value (hue counter-clockwise from +x, as in the original). */
function hueWheel(radius: number, dpr: number): HTMLCanvasElement {
  const key = `${radius}@${dpr}`;
  if (wheelCache?.key === key) return wheelCache.canvas;
  const n = Math.ceil(radius * 2 * dpr);
  const c = document.createElement('canvas');
  c.width = c.height = n;
  const ctx = c.getContext('2d')!;
  const id = ctx.createImageData(n, n);
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const x = (i + 0.5) / (n / 2) - 1;
      const y = 1 - (j + 0.5) / (n / 2);
      const r = Math.hypot(x, y);
      if (r > 1) continue;
      let h = Math.atan2(y, x) / (2 * Math.PI);
      if (h < 0) h += 1;
      const rgb = hsvToRgb([h, r, 1]);
      const k = (j * n + i) * 4;
      id.data[k] = rgb[0] * 255;
      id.data[k + 1] = rgb[1] * 255;
      id.data[k + 2] = rgb[2] * 255;
      id.data[k + 3] = Math.min(1, (1 - r) * radius * dpr) * 255; // anti-aliased rim
    }
  ctx.putImageData(id, 0, 0);
  wheelCache = { key, canvas: c };
  return c;
}

export function drawOverlay(ctx: CanvasRenderingContext2D, s: OverlayState): void {
  const { dpr } = s;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.lineCap = 'butt';

  // example edge (blue) in edge mode
  if (s.exampleEdge && s.exampleEdge.length > 1) {
    ctx.strokeStyle = 'rgb(51,51,179)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.exampleEdge.forEach((p, i) => {
      const [x, y] = s.toScreen(p);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // current stroke line ('t'), alternating black/white every 4 samples
  if (s.strokeLine && s.strokeLine.length > 1) {
    ctx.lineWidth = 1.5;
    for (let i = 0; i + 1 < s.strokeLine.length; i++) {
      ctx.strokeStyle = Math.floor(i / 4) % 2 ? '#fff' : '#000';
      ctx.beginPath();
      ctx.moveTo(...s.toScreen(s.strokeLine[i]));
      ctx.lineTo(...s.toScreen(s.strokeLine[i + 1]));
      ctx.stroke();
    }
  }

  // captured edge preview (perp band above tan band, x10 + 0.5 like the original)
  if (s.edgePreview) {
    const z = s.edgePreviewScale;
    let y = 64;
    ctx.imageSmoothingEnabled = false;
    for (const img of [s.edgePreview.perp, s.edgePreview.tan]) {
      const c = new OffscreenCanvas(img.width, img.height);
      c.getContext('2d')!.putImageData(img, 0, 0);
      const w = Math.min(img.width * z, ctx.canvas.width / dpr - 24);
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fillRect(10, y - 2, w + 4, img.height * z + 4);
      ctx.drawImage(c, 0, 0, w / z, img.height, 12, y, w, img.height * z);
      y += img.height * z + 10;
    }
  }

  // clone source markers
  if (s.mode === 'clone' && s.clonePoint) crosshair(ctx, s.toScreen(s.clonePoint), 6, '#ffd54a');
  if (s.cloneSource) crosshair(ctx, s.cloneSource, 6, '#ffd54a');

  // colour selector
  if (s.colorSelector) {
    const { hsv, valueMode, radius: R } = s.colorSelector;
    const pos: Vec2 = valueMode ? [hsv[2] * R, 0] : [Math.cos(hsv[0] * 2 * Math.PI) * hsv[1] * R, Math.sin(hsv[0] * 2 * Math.PI) * hsv[1] * R];
    const cx = s.mouse[0] - pos[0];
    const cy = s.mouse[1] + pos[1];
    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.6)';
    ctx.shadowBlur = 12;
    if (!valueMode) {
      ctx.drawImage(hueWheel(R, dpr), cx - R, cy - R, 2 * R, 2 * R);
    } else {
      const g = ctx.createLinearGradient(cx - R, 0, cx + R, 0);
      g.addColorStop(0, css(hsvToRgb([(hsv[0] + 0.5) % 1, hsv[1], 1])));
      g.addColorStop(0.5, '#000');
      g.addColorStop(1, css(hsvToRgb([hsv[0], hsv[1], 1])));
      ctx.fillStyle = g;
      ctx.fillRect(cx - R, cy - 0.1 * R, 2 * R, 0.2 * R);
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.font = '11px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('−1', cx - R, cy + 0.1 * R + 13);
      ctx.fillText('0', cx, cy + 0.1 * R + 13);
      ctx.fillText('+1', cx + R, cy + 0.1 * R + 13);
    }
    ctx.restore();
    crosshair(ctx, s.mouse, 4, '#000');
    return; // the brush cursor is hidden while picking
  }

  // cursor: crosshair + brush radius ticks
  if (s.mouseInside) {
    const [x, y] = s.mouse;
    const r = s.brushRadiusPx;
    crosshair(ctx, [x, y], 5);
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#000';
    ctx.beginPath();
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      ctx.moveTo(x + dx * r, y + dy * r);
      ctx.lineTo(x + dx * (r + 4), y + dy * (r + 4));
    }
    ctx.stroke();
    ctx.strokeStyle = '#fff';
    ctx.beginPath();
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      ctx.moveTo(x + dx * r - dy * 3, y + dy * r - dx * 3);
      ctx.lineTo(x + dx * r + dy * 3, y + dy * r + dx * 3);
    }
    ctx.stroke();
    if (r > 12) {
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}
