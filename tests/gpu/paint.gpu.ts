/**
 * WebGPU painting tests: stroke rasterisation, the five blend modes, the
 * clone and edge brushes, undo, and painting + integration end to end.
 */
import { blend, BLEND_MODES, BlendMode } from '../../src/paint/blend';
import { buildStrokeGeometry } from '../../src/paint/geometry';
import { GradientCanvas, readTexture } from '../../src/paint/painter';
import type { Vec2 } from '../../src/paint/stroke';
import { GpuSolver } from '../../src/solver/gpu';
import { Field } from '../../src/solver/reference';
import { randomField } from '../unit/exact';
import { assert, assertClose, assertLess, describe, test } from './harness';

const W = 64;
const H = 48;
const Y0 = 20; // stroke centre line
const R = 4; // brush radius
const COLOR: [number, number, number] = [0.8, -0.4, 0.3];

const line = (x0: number, x1: number, y: number): Vec2[] => {
  const p: Vec2[] = [];
  for (let x = x0; x <= x1; x++) p.push([x, y]);
  return p;
};

/** Paint-brush profile (render_paint_stroke): tent of half-width r/2, peak 1/size. */
const tent = (v: number) => Math.max(0, 1 - (2 * Math.abs(v)) / R) / R;

function fieldTex(f: Field) {
  return new Float32Array(f.data) as Float32Array<ArrayBuffer>;
}

function clampedBilinear(f: Field, tcx: number, tcy: number, c: number): number {
  const sx = tcx - 0.5;
  const sy = tcy - 0.5;
  const x0 = Math.floor(sx);
  const y0 = Math.floor(sy);
  const fx = sx - x0;
  const fy = sy - y0;
  const a = (1 - fx) * f.edge(x0, y0, c) + fx * f.edge(x0 + 1, y0, c);
  const b = (1 - fx) * f.edge(x0, y0 + 1, c) + fx * f.edge(x0 + 1, y0 + 1, c);
  return (1 - fy) * a + fy * b;
}

function paintStroke(canvas: GradientCanvas, points: Vec2[], mode: BlendMode, brush = canvas.paintBrush(COLOR, R)) {
  const enc = canvas.device.createCommandEncoder();
  canvas.pushSnapshot(enc);
  canvas.device.queue.submit([enc.finish()]);
  const enc2 = canvas.device.createCommandEncoder();
  canvas.renderStroke(enc2, buildStrokeGeometry(points, { radius: R, texRadius: 2, stepsAlong: 1 }), brush, mode);
  canvas.device.queue.submit([enc2.finish()]);
}

describe('gradient brush', () => {
  test('emits colour x tent profile along the stroke normal (add on a blank canvas)', async ({ device }) => {
    const canvas = new GradientCanvas(device, W, H);
    paintStroke(canvas, line(5, 58, Y0), 'add');
    const gx = new Field(W - 1, H, await readTexture(device, canvas.gx));
    const gy = new Field(W, H - 1, await readTexture(device, canvas.gy));
    for (let i = 15; i < 45; i++) {
      let colSum = 0;
      for (let j = 0; j < H - 1; j++) {
        // gy(i, j) sits at y = j + 1; the normal points up the screen (-y)
        const v = -(j + 1 - Y0);
        for (let c = 0; c < 3; c++) {
          const got = gy.data[gy.idx(i, j) + c];
          const want = -COLOR[c] * tent(v);
          assert(Math.abs(got - want) < 1e-6, `gy(${i},${j})[${c}] = ${got}, expected ${want}`);
        }
        colSum += gy.data[gy.idx(i, j)];
      }
      // integrated across the stroke the brush makes a step of colour/2, brighter above
      assert(Math.abs(colSum + COLOR[0] / 2) < 1e-6, `step across the stroke ${colSum}`);
      for (let j = 0; j < H; j++) assert(Math.abs(gx.data[gx.idx(i, j)]) < 1e-6, `gx(${i},${j}) should be 0`);
    }
    canvas.destroy();
  });

  test('leaves everything outside the stroke footprint untouched', async ({ device }) => {
    const base = { gx: randomField(W - 1, H, 1, 0.3), gy: randomField(W, H - 1, 2, 0.3) };
    const canvas = new GradientCanvas(device, W, H, fieldTex(base.gx), fieldTex(base.gy));
    paintStroke(canvas, line(10, 50, Y0), 'over');
    const gy = await readTexture(device, canvas.gy);
    for (let j = 0; j < H - 1; j++)
      for (let i = 0; i < W; i++) {
        const far = Math.abs(j + 1 - Y0) > R + 1.5 || i < 10 - R - 2 || i > 50 + R + 2;
        if (!far) continue;
        const k = (j * W + i) * 4;
        assert(gy[k] === Math.fround(base.gy.data[k]), `pixel (${i},${j}) changed`);
      }
    canvas.destroy();
  });
});

describe('blend modes on the GPU vs CPU model', () => {
  for (const mode of BLEND_MODES) {
    test(`${mode}`, async ({ device }) => {
      const base = { gx: randomField(W - 1, H, 3, 0.2), gy: randomField(W, H - 1, 4, 0.2) };
      // make sure the background is not in float32-rounding limbo
      const b32 = { gx: new Field(W - 1, H, new Float32Array(base.gx.data)), gy: new Field(W, H - 1, new Float32Array(base.gy.data)) };
      const canvas = new GradientCanvas(device, W, H, fieldTex(b32.gx), fieldTex(b32.gy));
      paintStroke(canvas, line(5, 58, Y0), mode);
      const gx = await readTexture(device, canvas.gx);
      const gy = await readTexture(device, canvas.gy);
      const got: number[] = [];
      const want: number[] = [];
      for (let i = 15; i < 45; i++) {
        for (let j = Y0 - R + 1; j <= Y0 + R - 2; j++) {
          // gx sample at (i+1, j+0.5); gy sample at (i+0.5, j+1)
          const vx = -(j + 0.5 - Y0);
          const vy = -(j + 1 - Y0);
          for (let c = 0; c < 3; c++) {
            const baseAtX: [number, number] = [b32.gx.data[b32.gx.idx(i, j) + c], clampedBilinear(b32.gy, i + 1, j, c)];
            const baseAtY: [number, number] = [clampedBilinear(b32.gx, i, j + 1, c), b32.gy.data[b32.gy.idx(i, j) + c]];
            want.push(blend(mode, baseAtX, [0, -COLOR[c] * tent(vx)])[0]);
            got.push(gx[(j * (W - 1) + i) * 4 + c]);
            want.push(blend(mode, baseAtY, [0, -COLOR[c] * tent(vy)])[1]);
            got.push(gy[(j * W + i) * 4 + c]);
          }
        }
      }
      assertClose(got, want, 2e-6, `${mode} blend`);
      canvas.destroy();
    });
  }
});

describe('clone brush', () => {
  test('copies source gradients at the clone offset (over)', async ({ device }) => {
    const base = { gx: randomField(W - 1, H, 5, 0.5), gy: randomField(W, H - 1, 6, 0.5) };
    const b32 = { gx: new Field(W - 1, H, new Float32Array(base.gx.data)), gy: new Field(W, H - 1, new Float32Array(base.gy.data)) };
    const canvas = new GradientCanvas(device, W, H, fieldTex(b32.gx), fieldTex(b32.gy));
    const pts = line(8, 40, 30);
    const offset: Vec2 = [13, -17];
    const enc = device.createCommandEncoder();
    canvas.pushSnapshot(enc);
    device.queue.submit([enc.finish()]);
    const enc2 = device.createCommandEncoder();
    canvas.renderStroke(enc2, buildStrokeGeometry(pts, { radius: R, texRadius: 0, stepsAlong: 0, cloneOffset: offset }), canvas.cloneBrush(), 'over');
    device.queue.submit([enc2.finish()]);
    const gx = new Field(W - 1, H, await readTexture(device, canvas.gx));
    const gy = new Field(W, H - 1, await readTexture(device, canvas.gy));
    for (let i = 12; i < 36; i++)
      for (let j = 30 - R + 1; j < 30 + R - 1; j++)
        for (let c = 0; c < 3; c++) {
          assert(Math.abs(gx.data[gx.idx(i, j) + c] - b32.gx.data[b32.gx.idx(i + 13, j - 17) + c]) < 1e-6, `gx(${i},${j}) = ${gx.data[gx.idx(i, j) + c]} want ${b32.gx.data[b32.gx.idx(i + 13, j - 17) + c]} [${[-1,0,1].map(d=>b32.gx.data[b32.gx.idx(i + 13 + d, j - 17) + c].toFixed(4)).join(' ')}] [${[-1,0,1].map(d=>b32.gx.data[b32.gx.idx(i + 13, j - 17+d) + c].toFixed(4)).join(' ')}]`);
          assert(Math.abs(gy.data[gy.idx(i, j) + c] - b32.gy.data[b32.gy.idx(i + 13, j - 17) + c]) < 1e-6, `gy(${i},${j})`);
        }
    canvas.destroy();
  });
});

describe('edge brush', () => {
  test('captures an edge and plays it back along a new stroke', async ({ device }) => {
    const canvas = new GradientCanvas(device, W, H);
    // image u = 0.1 y + 0.02 x (all channels)
    const img = Field.fill(W, H, (x, y, c) => (c === 3 ? 0 : 0.1 * y + 0.02 * x));
    const buf = device.createBuffer({ size: img.data.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, new Float32Array(img.data));
    const edge = canvas.captureEdge(buf, line(5, 45, 20), R);
    assert(edge, 'edge captured');
    assert(edge.steps === 40 && edge.perp.width === 40 && edge.perp.height === 2 * R - 1, 'perp texture size');
    assert(edge.tan.width === 39 && edge.tan.height === 2 * R, 'tan texture size');
    const perp = await readTexture(device, edge.perp);
    const tan = await readTexture(device, edge.tan);
    // rows advance along the normal (up the screen): u drops by 0.1 per row; along the stroke it rises 0.02 per pixel
    for (let k = 0; k < perp.length; k += 4) assert(Math.abs(perp[k] + 0.1) < 1e-5 && perp[k + 3] === 1, `perp ${perp[k]}`);
    for (let k = 0; k < tan.length; k += 4) assert(Math.abs(tan[k] - 0.02) < 1e-5, `tan ${tan[k]}`);

    // play back along y = 34 with 'over': the gradients of the original ramp reappear
    const enc = device.createCommandEncoder();
    canvas.pushSnapshot(enc);
    device.queue.submit([enc.finish()]);
    const enc2 = device.createCommandEncoder();
    const pts = line(4, 60, 34);
    canvas.renderStroke(enc2, buildStrokeGeometry(pts, { radius: R, texRadius: edge.radius, stepsAlong: edge.steps }), canvas.edgeBrush(edge), 'over');
    device.queue.submit([enc2.finish()]);
    const gx = new Field(W - 1, H, await readTexture(device, canvas.gx));
    const gy = new Field(W, H - 1, await readTexture(device, canvas.gy));
    for (let i = 10; i < 55; i++)
      for (let j = 34 - 2; j <= 34 + 1; j++) {
        assert(Math.abs(gy.data[gy.idx(i, j)] - 0.1) < 1e-5, `played-back gy(${i},${j}) = ${gy.data[gy.idx(i, j)]}`);
        assert(Math.abs(gx.data[gx.idx(i, j)] - 0.02) < 1e-5, `played-back gx(${i},${j}) = ${gx.data[gx.idx(i, j)]}`);
      }
    canvas.destroy();
  });
});

describe('history', () => {
  test('undo restores the exact previous gradients; revert returns to the original', async ({ device }) => {
    const base = { gx: randomField(W - 1, H, 7, 0.5), gy: randomField(W, H - 1, 8, 0.5) };
    const canvas = new GradientCanvas(device, W, H, fieldTex(base.gx), fieldTex(base.gy));
    const orig = await readTexture(device, canvas.gx);
    paintStroke(canvas, line(5, 50, 10), 'add');
    const after1 = await readTexture(device, canvas.gx);
    paintStroke(canvas, line(5, 50, 30), 'over');
    let enc = device.createCommandEncoder();
    assert(canvas.undo(enc), 'undo 1');
    device.queue.submit([enc.finish()]);
    assertClose(await readTexture(device, canvas.gx), after1, 0, 'after undoing stroke 2');
    enc = device.createCommandEncoder();
    canvas.revert(enc);
    device.queue.submit([enc.finish()]);
    assertClose(await readTexture(device, canvas.gx), orig, 0, 'after revert');
    enc = device.createCommandEncoder();
    canvas.undo(enc); // undo the revert
    device.queue.submit([enc.finish()]);
    assertClose(await readTexture(device, canvas.gx), after1, 0, 'after undoing revert');
    enc = device.createCommandEncoder();
    canvas.undo(enc);
    assert(!canvas.undo(enc), 'history exhausted');
    device.queue.submit([enc.finish()]);
    assertClose(await readTexture(device, canvas.gx), orig, 0, 'back to start');
    canvas.destroy();
  });
});

describe('paint + integrate end to end', () => {
  test('a long horizontal gradient stroke makes the area above brighter by ~colour/2', async ({ device, note }) => {
    const w = 129;
    const h = 129;
    const canvas = new GradientCanvas(device, w, h);
    const solver = new GpuSolver(device, w, h);
    solver.setGradientTextures(canvas.gx, canvas.gy);
    solver.setTargetMean([0.5, 0.5, 0.5]);
    const enc = device.createCommandEncoder();
    canvas.pushSnapshot(enc);
    device.queue.submit([enc.finish()]);
    const enc2 = device.createCommandEncoder();
    canvas.renderStroke(enc2, buildStrokeGeometry(line(-10, w + 10, 64), { radius: R, texRadius: 2, stepsAlong: 1 }), canvas.paintBrush([0.6, 0.6, 0.6], R), 'add');
    device.queue.submit([enc2.finish()]);
    solver.run(20, true);
    const u = new Field(w, h, await solver.readSolution());
    const above = u.data[u.idx(64, 40)];
    const below = u.data[u.idx(64, 90)];
    note(`above ${above.toFixed(4)}, below ${below.toFixed(4)}`);
    assertLess(Math.abs(above - below - 0.3), 1e-3, 'step across a full-width stroke');
    assertLess(Math.abs((above + below) / 2 - 0.5), 5e-3, 'symmetric about the target mean');
    solver.destroy();
    canvas.destroy();
  });
});

describe('pen pressure', () => {
  async function strokeWith(device: GPUDevice, opts: { radii?: number[]; strengths?: number[] }) {
    const canvas = new GradientCanvas(device, W, H);
    const pts = line(5, 58, Y0);
    const enc = device.createCommandEncoder();
    canvas.pushSnapshot(enc);
    canvas.renderStroke(enc, buildStrokeGeometry(pts, { radius: R, texRadius: 2, stepsAlong: 1, ...opts }), canvas.paintBrush(COLOR, R), 'add');
    device.queue.submit([enc.finish()]);
    const gy = new Field(W, H - 1, await readTexture(device, canvas.gy));
    canvas.destroy();
    return { gy, pts };
  }

  test('strength scales the brush gradient exactly', async ({ device }) => {
    const n = 54;
    const { gy } = await strokeWith(device, { strengths: Array(n).fill(0.4) });
    for (let i = 15; i < 45; i++)
      for (let j = 0; j < H - 1; j++) {
        const want = -0.4 * COLOR[0] * tent(-(j + 1 - Y0));
        assert(Math.abs(gy.data[gy.idx(i, j)] - want) < 1e-6, `gy(${i},${j}) = ${gy.data[gy.idx(i, j)]}, expected ${want}`);
      }
  });

  test('strength varying along the stroke is interpolated between samples', async ({ device }) => {
    const pts = line(5, 58, Y0);
    const strengths = pts.map((p) => (p[0] - 5) / 53); // 0 -> 1 left to right
    const { gy } = await strokeWith(device, { strengths });
    let prev = 0;
    for (let i = 10; i < 54; i++) {
      const colSum = -[...Array(H - 1).keys()].reduce((a, j) => a + gy.data[gy.idx(i, j)], 0);
      const want = (COLOR[0] / 2) * ((i + 0.5 - 5) / 53); // gy columns sit at x = i + 0.5
      assert(Math.abs(colSum - want) < 2e-3, `column ${i}: step ${colSum}, expected ${want}`);
      assert(colSum > prev, 'monotone along the stroke');
      prev = colSum;
    }
  });

  test('size mode changes the width but keeps the step across a gradient-brush stroke', async ({ device }) => {
    const n = 54;
    // pressed lightly: thinner stroke (tent half-width radius/2 = 1 px, so pixel sampling is exact)
    const radius = 2;
    const { gy } = await strokeWith(device, { radii: Array(n).fill(radius), strengths: Array(n).fill(R / radius) });
    for (let i = 15; i < 45; i++) {
      let sum = 0;
      let widest = 0;
      for (let j = 0; j < H - 1; j++) {
        const v = gy.data[gy.idx(i, j)];
        sum += v;
        if (Math.abs(v) > 1e-7) widest = Math.max(widest, Math.abs(j + 1 - Y0));
      }
      assert(Math.abs(sum + COLOR[0] / 2) < 1e-5, `step ${sum} should stay ${-COLOR[0] / 2}`);
      assert(widest < R / 2, `profile narrower than the full-pressure tent (${widest})`);
    }
  });
});
